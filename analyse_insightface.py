import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import math
import base64

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def rel_dist(p1, p2, ref):
    return euclidean(p1, p2) / ref

def symmetry(nose, pL, pR):
    return abs(euclidean(nose, pL) - euclidean(nose, pR)) / ((euclidean(nose, pL) + euclidean(nose, pR)) / 2) * 100

def angle(p1, p2, p3):
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

def image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

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

# Charger InsightFace (1 seule fois au démarrage)
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def analyse_image_insightface(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    faces = app.get(image_rgb)

    if not faces:
        return {"error": "Aucun visage détecté."}

    face = faces[0]
    pts = face.landmark_3d_68.tolist()

    # Landmarks clés
    leye, reye = pts[36], pts[45]
    nose = pts[30]
    chin = pts[8]
    mouth_left, mouth_right = pts[48], pts[54]
    top_lip, bottom_lip = pts[51], pts[57]
    left_brow, right_brow = pts[17], pts[26]
    left_cheek, right_cheek = pts[1], pts[15]
    top_head = pts[27]  # approximation
    mid_forehead = pts[24]  # entre sourcils

    face_width = euclidean(pts[0], pts[16])
    face_height = euclidean(pts[8], pts[27])

    mesures = {
        "face_width_px": round(face_width, 1),
        "face_height_px": round(face_height, 1),
        "ratio_largeur_hauteur": round(face_width / face_height, 3),
        "interoculaire_ratio": round(rel_dist(leye, reye, face_width), 3),
        "largeur_nez_ratio": round(rel_dist(pts[31], pts[35], face_width), 3),
        "largeur_bouche_ratio": round(rel_dist(mouth_left, mouth_right, face_width), 3),
        "hauteur_front_ratio": round(rel_dist(top_head, mid_forehead, face_height), 3),
        "distance_yeux_sourcils_ratio": round(rel_dist(((left_brow[0]+right_brow[0])/2, (left_brow[1]+right_brow[1])/2), ((leye[0]+reye[0])/2, (leye[1]+reye[1])/2), face_height), 3),
        "distance_nez_bouche_ratio": round(rel_dist(nose, top_lip, face_height), 3),
        "distance_nez_menton_ratio": round(rel_dist(nose, chin, face_height), 3),
        "distance_bouche_menton_ratio": round(rel_dist(bottom_lip, chin, face_height), 3),
        "tiers_superieur": round(rel_dist(top_head, mid_forehead, face_height), 3),
        "tiers_milieu": round(rel_dist(mid_forehead, nose, face_height), 3),
        "tiers_inferieur": round(rel_dist(nose, chin, face_height), 3),
        "asym_yeux": round(symmetry(nose, leye, reye), 2),
        "asym_bouche": round(symmetry(nose, mouth_left, mouth_right), 2),
        "asym_joues": round(symmetry(nose, left_cheek, right_cheek), 2),
        "asym_sourcils": round(symmetry(nose, left_brow, right_brow), 2),
        "asym_total_moyenne": round(np.mean([
            symmetry(nose, leye, reye),
            symmetry(nose, mouth_left, mouth_right),
            symmetry(nose, left_cheek, right_cheek),
            symmetry(nose, left_brow, right_brow)
        ]), 2),
        "angle_mandibule": round(angle(left_cheek, chin, right_cheek), 2),
        "angle_menton_nez_front": round(angle(chin, nose, top_head), 2),
        "courbure_mandibule": round(rel_dist(pts[4], pts[12], face_width), 3),
        "largeur_pommettes": round(rel_dist(left_cheek, right_cheek, face_width), 3),
        "profondeur_joue_gauche": round(rel_dist(pts[4], left_cheek, face_width), 3),
        "profondeur_joue_droite": round(rel_dist(pts[12], right_cheek, face_width), 3),
    }

    # Mesures pour analyse secondaire
    eye_opening_avg = round((rel_dist(pts[38], pts[40], face_height) + rel_dist(pts[44], pts[46], face_height)) / 2, 3)
    brow_dist_avg = round((rel_dist(pts[17], pts[38], face_height) + rel_dist(pts[26], pts[44], face_height)) / 2, 3)
    mouth_angle = angle(mouth_left, bottom_lip, mouth_right)

    age = estimate_age(eye_opening_avg, brow_dist_avg)
    tensions = detect_tensions(eye_opening_avg, mouth_angle)
    ethnicity = estimate_ethnicity(mesures["largeur_nez_ratio"] / mesures["distance_nez_menton_ratio"], mesures["largeur_bouche_ratio"], mesures["largeur_pommettes"], mesures["interoculaire_ratio"])
    modifications = detect_modifications(mesures["largeur_nez_ratio"] / mesures["distance_nez_menton_ratio"], mesures["largeur_bouche_ratio"], mesures["largeur_pommettes"])

    for pt in pts:
        cv2.circle(image, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

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
        "image_annotee_base64": image_to_base64(image)
    }
