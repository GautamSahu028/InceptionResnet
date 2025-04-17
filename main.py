from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from face_recognizer import FaceRecognizer
from PIL import Image
import io

app = FastAPI()

# Enable CORS for localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recognizer = FaceRecognizer()
recognizer.load_embeddings("embeddings.pth")

@app.post("/recognize")
async def recognize_faces(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))

    recognizer.unknown_embeddings = {}
    recognizer.detect_faces(img)
    results = []

    for key, test_emb in recognizer.unknown_embeddings.items():
        predicted_name, similarity_score = recognizer.predict_name(test_emb)
        results.append({
            "face_id": key,
            "predicted_name": predicted_name,
            "similarity_score": round(similarity_score, 3)
        })

    return JSONResponse(content={"results": results})
