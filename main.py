from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import mediapipe as mp
import math
from io import BytesIO

app = FastAPI()

# ==== Fonction utilitaire (MediaPipe + mesures) ====
def analyse_image(image_bytes):
    # Lecture de l'image depuis les bytes
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # Initialisation de MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    # Détection
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return {"error": "Aucun visage détecté"}

    landmarks = results.multi_face_landmarks[0].landmark
    landmarks_points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    def dist(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # Dimensions visage
    x_coords = [pt[0] for pt in landmarks_points]
    y_coords = [pt[1] for pt in landmarks_points]
    face_width = max(x_coords) - min(x_coords)
    face_height = max(y_coords) - min(y_coords)

    # Calculs
    interocular = dist(landmarks_points[133], landmarks_points[362]) / face_width
    nose_width = dist(landmarks_points[98], landmarks_points[327]) / face_width
    angle_jaw = math.degrees(math.atan2(
        landmarks_points[152][1] - landmarks_points[234][1],
        landmarks_points[152][0] - landmarks_points[234][0]
    ))
    if angle_jaw < 0: angle_jaw += 360
    eyebrow_y = (landmarks_points[55][1] + landmarks_points[285][1]) / 2
    forehead_height = abs(landmarks_points[10][1] - eyebrow_y)
    forehead_ratio = forehead_height / face_height

    nose_tip = landmarks_points[1]
    def sym(pL, pR): return abs(dist(nose_tip, pL) - dist(nose_tip, pR)) / ((dist(nose_tip, pL) + dist(nose_tip, pR)) / 2) * 100
    asym = (sym(landmarks_points[133], landmarks_points[362]) +
            sym(landmarks_points[61], landmarks_points[291]) +
            sym(landmarks_points[234], landmarks_points[454])) / 3

    # Résultat
    return {
        "distance_interoculaire": round(interocular, 3),
        "largeur_nez": round(nose_width, 3),
        "angle_machoire_gauche": round(angle_jaw, 1),
        "ratio_front_visage": round(forehead_ratio, 2),
        "asymetrie_faciale_totale": round(asym, 1)
    }

# ==== Endpoint API ====
@app.post("/analyse-visage")
async def analyse_visage(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = analyse_image(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ==== Démarrage en local ====
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
