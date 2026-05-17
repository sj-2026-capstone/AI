from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import shutil
import uuid
import httpx

from inference import predict_image, generate_grad_cam

app = FastAPI()

UPLOAD_DIR = "temp_uploads"
GRAD_CAM_DIR = "grad_cam_images"
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GRAD_CAM_DIR, exist_ok=True)

app.mount("/grad-cam-images", StaticFiles(directory=GRAD_CAM_DIR), name="grad_cam_images")


class AnalyzeRequest(BaseModel):
    inspectionId: str
    imageUrl: str
    callbackUrl: str


async def run_analysis(image_url: str, callback_url: str):
    ext = os.path.splitext(image_url.split("?")[0])[1] or ".jpg"
    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(
                image_url,
                headers={"X-Service-Key": "d74d235b1bbe0b3e49f53d7a9cca2766704abb1d2e1892b818f1217690ebc933"},
            )
            response.raise_for_status()
            with open(temp_path, "wb") as f:
                f.write(response.content)

        result = predict_image(temp_path)

        grad_cam_url = None
        try:
            target_class_idx = 1 if result["prediction"] == "DEFECT" else 0
            grad_cam_filename = f"{uuid.uuid4()}.jpg"
            grad_cam_path = os.path.join(GRAD_CAM_DIR, grad_cam_filename)
            generate_grad_cam(temp_path, grad_cam_path, target_class_idx=target_class_idx)
            grad_cam_url = f"{BASE_URL}/grad-cam-images/{grad_cam_filename}"
        except Exception as e:
            print(f"[WARN] Grad-CAM generation failed: {e}")

        callback_payload = {
            "hasDefect": result["prediction"] == "DEFECT",
            "defectType": None,
            "resultNote": f"prediction={result['prediction']}, confidence={result['confidence']:.4f}, defect_prob={result['defect_probability']:.4f}",
            "gradCamImageUrl": grad_cam_url,
        }

        async with httpx.AsyncClient() as client:
            await client.post(
                callback_url,
                json=callback_payload,
                headers={"X-Service-Key": "d74d235b1bbe0b3e49f53d7a9cca2766704abb1d2e1892b818f1217690ebc933"},
            )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/")
def root():
    return {"message": "AI Server Running"}


@app.post("/analyze")
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis, request.imageUrl, request.callbackUrl)
    return {"status": "accepted", "inspectionId": request.inspectionId}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1]
    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predict_image(temp_path)

        return {"filename": file.filename, **result}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
