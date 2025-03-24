from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from analyse_insightface import analyse_image_insightface
from analyse_mediapipe import analyse_image_mediapipe
import cv2
import numpy as np
import base64

app = FastAPI(
    title="API Analyse Faciale Biométrique",
    description="Analyse morphologique détaillée du visage via InsightFace ou MediaPipe.",
    version="1.0"
)

def image_base64_to_cv2(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def concat_images_side_by_side(img1, img2):
    height = max(img1.shape[0], img2.shape[0])
    img1_resized = cv2.resize(img1, (img1.shape[1], height))
    img2_resized = cv2.resize(img2, (img2.shape[1], height))
    return cv2.hconcat([img1_resized, img2_resized])

@app.post("/analyse-visage")
async def analyse_visage(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = analyse_image_insightface(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/analyse-mediapipe")
async def analyse_mediapipe(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = analyse_image_mediapipe(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/analyse-combinee")
async def analyse_combinee(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        insight_result = analyse_image_insightface(image_bytes)
        mediapipe_result = analyse_image_mediapipe(image_bytes)

        if "error" in insight_result:
            return JSONResponse(content={"error": "InsightFace: " + insight_result["error"]}, status_code=400)
        if "error" in mediapipe_result:
            return JSONResponse(content={"error": "MediaPipe: " + mediapipe_result["error"]}, status_code=400)

        img1 = image_base64_to_cv2(insight_result["image_annotee_base64"])
        img2 = image_base64_to_cv2(mediapipe_result["image_mediapipe_base64"])
        fused_img = concat_images_side_by_side(img1, img2)
        fused_base64 = image_to_base64(fused_img)

        return JSONResponse(content={
            "image_fusion_base64": fused_base64,
            "resultat_insightface": insight_result,
            "resultat_mediapipe": mediapipe_result
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
