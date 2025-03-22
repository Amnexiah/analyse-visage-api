from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import mediapipe as mp
import math

app = FastAPI()

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def angle(p1, p2, p3):
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

def rel_dist(p1, p2, ref):
    return euclidean(p1, p2) / ref

def symmetry(nose, pL, pR):
    return abs(euclidean(nose, pL) - euclidean(nose, pR)) / ((euclidean(nose, pL) + euclidean(nose, pR)) / 2) * 100

def analyse_image(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return {"error": "Aucun visage détecté"}

    lm = results.multi_face_landmarks[0].landmark
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]

    face_width = max(p[0] for p in pts) - min(p[0] for p in pts)
    face_height = max(p[1] for p in pts) - min(p[1] for p in pts)

    # Points clés (MediaPipe 468)
    leye, reye = pts[133], pts[362]
    nose_tip, nose_base = pts[1], pts[2]
    chin = pts[152]
    top_head = pts[10]
    mid_forehead = pts[168]
    left_cheek, right_cheek = pts[234], pts[454]
    mouth_left, mouth_right = pts[61], pts[291]
    top_lip, bottom_lip = pts[13], pts[14]
    left_brow, right_brow = pts[105], pts[334]
    brow_center = ((left_brow[0]+right_brow[0])//2, (left_brow[1]+right_brow[1])//2)
    eye_center = ((leye[0]+reye[0])//2, (leye[1]+reye[1])//2)

    # === Mesures biométriques avancées ===
    mesures = {
        "face_width_px": face_width,
        "face_height_px": face_height,
        "ratio_largeur_hauteur": round(face_width / face_height, 3),

        # Distances clés relatives
        "interoculaire_ratio": round(rel_dist(leye, reye, face_width), 3),
        "largeur_nez_ratio": round(rel_dist(pts[98], pts[327], face_width), 3),
        "largeur_bouche_ratio": round(rel_dist(mouth_left, mouth_right, face_width), 3),
        "hauteur_front_ratio": round(rel_dist(top_head, mid_forehead, face_height), 3),
        "distance_yeux_sourcils_ratio": round(rel_dist(brow_center, eye_center, face_height), 3),
        "distance_nez_bouche_ratio": round(rel_dist(nose_tip, top_lip, face_height), 3),
        "distance_nez_menton_ratio": round(rel_dist(nose_tip, chin, face_height), 3),
        "distance_bouche_menton_ratio": round(rel_dist(bottom_lip, chin, face_height), 3),

        # Règle des tiers (front / nez / menton)
        "tiers_superieur": round(rel_dist(top_head, mid_forehead, face_height), 3),
        "tiers_milieu": round(rel_dist(mid_forehead, nose_tip, face_height), 3),
        "tiers_inferieur": round(rel_dist(nose_tip, chin, face_height), 3),

        # Asymétries régionales
        "asym_yeux": round(symmetry(nose_tip, leye, reye), 2),
        "asym_bouche": round(symmetry(nose_tip, mouth_left, mouth_right), 2),
        "asym_joues": round(symmetry(nose_tip, left_cheek, right_cheek), 2),
        "asym_sourcils": round(symmetry(nose_tip, left_brow, right_brow), 2),
        "asym_total_moyenne": round(
            np.mean([
                symmetry(nose_tip, leye, reye),
                symmetry(nose_tip, mouth_left, mouth_right),
                symmetry(nose_tip, left_cheek, right_cheek),
                symmetry(nose_tip, left_brow, right_brow)
            ]), 2),

        # Angles de structure
        "angle_mandibule": round(angle(left_cheek, chin, right_cheek), 2),
        "angle_menton_nez_front": round(angle(chin, nose_tip, top_head), 2),

        # Indices biométriques complexes
        "golden_ratio_approx": round(rel_dist(nose_tip, top_lip, face_height) / rel_dist(top_lip, chin, face_height), 2),
        "vertical_index": round(rel_dist(top_head, chin, face_height), 2),
        "horizontal_index": round(rel_dist(left_cheek, right_cheek, face_width), 2)
    }

    return {
        "mesures_biometriques": mesures
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

# ==== Lancement local ====

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
