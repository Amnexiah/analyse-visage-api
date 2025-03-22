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

def avg_y(pts):
    return sum(p[1] for p in pts) / len(pts)

def estimate_age(proportions):
    # Approximations basées sur tendances générales (très simplifié)
    ratio = proportions["rapport_hauteur_largeur"]
    eye_opening = proportions["ouverture_moyenne_yeux"]
    brow_dist = proportions["distance_oeil_sourcil_moyenne"]
    if eye_opening < 0.23 and brow_dist < 0.06:
        return "40-60+"
    elif eye_opening < 0.26:
        return "25-40"
    else:
        return "15-25"

def detect_tensions(paupiere_h, levre_angle):
    return {
        "paupiere_basse": paupiere_h < 0.22,
        "commissure_tendue": levre_angle < 155
    }

def estimate_ethnicity(ratios):
    # Approche empirique - absolument pas définitive ni réaliste
    if ratios["nez_ratio"] > 1.1 and ratios["pommette_ratio"] > 0.7:
        return "afro-descendant (approximatif)"
    elif ratios["nez_ratio"] < 0.9 and ratios["oeil_distance_ratio"] > 0.32:
        return "asiatique (approximatif)"
    elif ratios["nez_ratio"] < 1.1 and ratios["bouche_ratio"] > 0.35:
        return "caucasien / méditerranéen (approximatif)"
    else:
        return "inclassable / mixte"

def detect_modifications(ratios):
    # Détection de ratios hors norme (chirurgie possible)
    anomalies = []
    if ratios["nez_ratio"] < 0.6:
        anomalies.append("nez anormalement court ou modifié")
    if ratios["bouche_ratio"] > 0.6:
        anomalies.append("bouche excessivement large")
    if ratios["pommette_ratio"] > 0.85:
        anomalies.append("pommettes artificiellement saillantes")
    return anomalies

def analyse_image(image_bytes):
    # ... (code principal identique que précédemment jusqu'à calculs de mesures)

    # Simulations des mesures extraites pour démo (dans ton script final, remplace avec les vraies)
    rapport_hauteur_largeur = 0.88
    ouverture_moyenne_yeux = 0.245
    distance_oeil_sourcil_moyenne = 0.058
    nez_ratio = 1.05
    bouche_ratio = 0.38
    pommette_ratio = 0.73
    oeil_distance_ratio = 0.31
    levre_angle = 152

    proportions = {
        "rapport_hauteur_largeur": rapport_hauteur_largeur,
        "ouverture_moyenne_yeux": ouverture_moyenne_yeux,
        "distance_oeil_sourcil_moyenne": distance_oeil_sourcil_moyenne
    }

    ratios = {
        "nez_ratio": nez_ratio,
        "bouche_ratio": bouche_ratio,
        "pommette_ratio": pommette_ratio,
        "oeil_distance_ratio": oeil_distance_ratio
    }

    age_estime = estimate_age(proportions)
    tensions = detect_tensions(ouverture_moyenne_yeux, levre_angle)
    ethnie_estimee = estimate_ethnicity(ratios)
    anomalies = detect_modifications(ratios)

    return {
        "estimation_age_facial": age_estime,
        "tensions_musculaires": tensions,
        "estimation_ethnie_morphologique": ethnie_estimee,
        "anomalies_possibles": anomalies
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
