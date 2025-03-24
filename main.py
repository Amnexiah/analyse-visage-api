from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from analyse_insightface import analyse_image_insightface
from analyse_mediapipe import analyse_image_mediapipe

app = FastAPI(
    title="API Analyse Faciale Biométrique",
    description="Analyse morphologique détaillée du visage via InsightFace ou MediaPipe.",
    version="1.0"
)

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
