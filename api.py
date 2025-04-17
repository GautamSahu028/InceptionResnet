from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from face_recognizer import FaceRecognizer
from PIL import Image
import io
import os

app = FastAPI()
recognizer = FaceRecognizer()
recognizer.load_embeddings("embeddings.pth")

@app.post("/recognize")
async def recognize_faces(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))  # <-- fix here

    recognizer.unknown_embeddings = {}  # reset previous unknowns
    recognizer.detect_faces(img)  # use detect_faces directly on Image
    results = []

    for key, test_emb in recognizer.unknown_embeddings.items():
        predicted_name, similarity_score = recognizer.predict_name(test_emb)
        results.append({
            "face_id": key,
            "predicted_name": predicted_name,
            "similarity_score": round(similarity_score, 3)
        })

    return JSONResponse(content={"results": results})
