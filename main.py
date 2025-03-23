from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import mediapipe as mp
import math
import base64
from io import BytesIO

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

def avg_y(pts):
    return sum(p[1] for p in pts) / len(pts)

def estimate_age(opening_ratio, brow_dist):
    if opening_ratio < 0.23 and brow_dist < 0.06:
        return "40-60+"
    elif opening_ratio < 0.26:
        return "25-40"
    else:
        return "15-25"

def detect_tensions(eye_opening, mouth_angle):
    return {
        "paupiere_basse": bool(eye_opening < 0.22),
        "commissure_tendue": bool(mouth_angle < 155)
    }

def estimate_ethnicity(nose_ratio, mouth_ratio, cheek_ratio, eye_spacing_ratio):
    if nose_ratio > 1.1 and cheek_ratio > 0.7:
        return "afro-descendant (approximatif)"
    elif nose_ratio < 0.9 and eye_spacing_ratio > 0.32:
        return "asiatique (approximatif)"
    elif nose_ratio < 1.1 and mouth_ratio > 0.35:
        return "caucasien / méditerranéen (approximatif)"
    else:
        return "inclassable / mixte"

def detect_modifications(nose_ratio, mouth_ratio, cheek_ratio):
    anomalies = []
    if nose_ratio < 0.6:
        anomalies.append("nez anormalement court ou modifié")
    if mouth_ratio > 0.6:
        anomalies.append("bouche excessivement large")
    if cheek_ratio > 0.85:
        anomalies.append("pommettes artificiellement saillantes")
    return anomalies

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

    # Dessiner les points sur une copie de l'image
    image_copy = image.copy()
    for pt in pts:
        cv2.circle(image_copy, pt, 1, (0, 255, 0), -1)
    _, buffer = cv2.imencode('.jpg', image_copy)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    face_width = max(p[0] for p in pts) - min(p[0] for p in pts)
    face_height = max(p[1] for p in pts) - min(p[1] for p in pts)

    leye, reye = pts[133], pts[362]
    top_head = pts[10]
    mid_forehead = pts[168]
    mouth_left, mouth_right = pts[61], pts[291]
    top_lip, bottom_lip = pts[13], pts[14]
    nose_tip = pts[1]
    chin = pts[152]
    left_cheek, right_cheek = pts[234], pts[454]
    left_brow, right_brow = pts[105], pts[334]

    mesures = {
        "face_width_px": face_width,
        "face_height_px": face_height,
        "ratio_largeur_hauteur": round(face_width / face_height, 3),
        "interoculaire_ratio": round(rel_dist(leye, reye, face_width), 3),
        "largeur_nez_ratio": round(rel_dist(pts[98], pts[327], face_width), 3),
        "largeur_bouche_ratio": round(rel_dist(mouth_left, mouth_right, face_width), 3),
        "hauteur_front_ratio": round(rel_dist(top_head, mid_forehead, face_height), 3),
        "distance_yeux_sourcils_ratio": round(rel_dist(((left_brow[0]+right_brow[0])//2, (left_brow[1]+right_brow[1])//2), ((leye[0]+reye[0])//2, (leye[1]+reye[1])//2), face_height), 3),
        "distance_nez_bouche_ratio": round(rel_dist(nose_tip, top_lip, face_height), 3),
        "distance_nez_menton_ratio": round(rel_dist(nose_tip, chin, face_height), 3),
        "distance_bouche_menton_ratio": round(rel_dist(bottom_lip, chin, face_height), 3),
        "tiers_superieur": round(rel_dist(top_head, mid_forehead, face_height), 3),
        "tiers_milieu": round(rel_dist(mid_forehead, nose_tip, face_height), 3),
        "tiers_inferieur": round(rel_dist(nose_tip, chin, face_height), 3),
        "asym_yeux": round(symmetry(nose_tip, leye, reye), 2),
        "asym_bouche": round(symmetry(nose_tip, mouth_left, mouth_right), 2),
        "asym_joues": round(symmetry(nose_tip, left_cheek, right_cheek), 2),
        "asym_sourcils": round(symmetry(nose_tip, left_brow, right_brow), 2),
        "asym_total_moyenne": round(np.mean([
            symmetry(nose_tip, leye, reye),
            symmetry(nose_tip, mouth_left, mouth_right),
            symmetry(nose_tip, left_cheek, right_cheek),
            symmetry(nose_tip, left_brow, right_brow)
        ]), 2),
        "angle_mandibule": round(angle(left_cheek, chin, right_cheek), 2),
        "angle_menton_nez_front": round(angle(chin, nose_tip, top_head), 2),
        "courbure_mandibule": round(rel_dist(pts[127], pts[356], face_width), 3),
        "largeur_pommettes": round(rel_dist(left_cheek, right_cheek, face_width), 3),
        "profondeur_joue_gauche": round(rel_dist(pts[50], pts[234], face_width), 3),
        "profondeur_joue_droite": round(rel_dist(pts[280], pts[454], face_width), 3)
    }

    eye_opening_avg = round((rel_dist(pts[159], pts[145], face_height) + rel_dist(pts[386], pts[374], face_height)) / 2, 3)
    brow_dist_avg = round((rel_dist(pts[105], pts[159], face_height) + rel_dist(pts[334], pts[386], face_height)) / 2, 3)
    mouth_angle = angle(mouth_left, bottom_lip, mouth_right)

    age = estimate_age(eye_opening_avg, brow_dist_avg)
    tensions = detect_tensions(eye_opening_avg, mouth_angle)
    ethnicity = estimate_ethnicity(mesures["largeur_nez_ratio"] / mesures["distance_nez_menton_ratio"], mesures["largeur_bouche_ratio"], mesures["largeur_pommettes"], mesures["interoculaire_ratio"])
    modifications = detect_modifications(mesures["largeur_nez_ratio"] / mesures["distance_nez_menton_ratio"], mesures["largeur_bouche_ratio"], mesures["largeur_pommettes"])

    return {
        "mesures_biometriques": mesures,
        "analyses_secondaires": {
            "estimation_age_facial": age,
            "tensions_musculaires": tensions,
            "estimation_ethnie_morphologique": ethnicity,
            "anomalies_possibles": modifications
        },
        "vecteur_debug": {
            "eye_opening_avg": eye_opening_avg,
            "brow_dist_avg": brow_dist_avg,
            "mouth_angle": round(mouth_angle, 2)
        },
        "image_annotee_base64": image_base64
    }

@app.post("/analyse-visage")
async def analyse_visage(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = analyse_image(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
