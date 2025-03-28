import cv2
import numpy as np
import mediapipe as mp
import base64

def image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def analyse_image_mediapipe(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return {"error": "Aucun visage détecté par MediaPipe."}

        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                cv2.circle(annotated_image, (x, y), 1, (255, 0, 0), -1)

        return {
            "image_mediapipe_base64": image_to_base64(annotated_image),
            "nombre_points_detectes": len(results.multi_face_landmarks[0].landmark)
        }
