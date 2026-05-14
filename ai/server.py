from fastapi import FastAPI, UploadFile, File
import os
import shutil
import uuid

app = FastAPI()

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"message": "AI Server Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1]
    temp_filename = f"{uuid.uuid4()}{ext}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "filename": file.filename,
            "saved_path": temp_path,
            "message": "Image received successfully"
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
