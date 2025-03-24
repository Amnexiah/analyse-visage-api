import cv2
import numpy as np
import dlib
import base64
import os

predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.isfile(predictor_path):
    raise FileNotFoundError(f"Le fichier {predictor_path} est requis pour utiliser dlib.")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def analyse_image_dlib(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if not faces:
        return {"error": "Aucun visage détecté par dlib."}

    face = faces[0]
    landmarks = predictor(gray, face)

    pts = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]

    left_jaw = pts[0]
    right_jaw = pts[16]
    chin = pts[8]
    top_nose = pts[27]
    top_forehead = (pts[27][0], pts[27][1] - int(abs(pts[8][1] - pts[27][1]) * 0.5))

    face_width = euclidean(left_jaw, right_jaw)
    face_height = euclidean(chin, top_forehead)

    mesures = {
        "largeur_visage_px": round(face_width, 1),
        "hauteur_visage_px": round(face_height, 1),
        "ratio_largeur_hauteur": round(face_width / face_height, 3)
    }

    # Visualisation
    key_points = [left_jaw, right_jaw, chin, top_nose, top_forehead]
    for pt in key_points:
        cv2.circle(image, pt, 3, (0, 255, 0), -1)
    cv2.line(image, left_jaw, right_jaw, (255, 0, 0), 1)
    cv2.line(image, chin, top_forehead, (255, 0, 0), 1)

    return {
        "mesures_dlib": mesures,
        "image_dlib_base64": image_to_base64(image)
    }
