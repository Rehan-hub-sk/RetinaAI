from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from model_service import predict_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"status": "RetinaAI backend running"}


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    patientName: str = Form(""),
    patientId: str = Form(""),
    age: str = Form(""),
    gender: str = Form(""),
    diabetesDuration: str = Form(""),
    location: str = Form("")
):
    image_bytes = await image.read()

    result = predict_image(image_bytes)

    label = result["label"]
    confidence = result["confidence"]

    return {
        "status": "success",
        "prediction": label,
        "confidence": confidence,
        "risk": "AI model prediction",
        "assessment": "Prototype AI screening result",
        "explanation": (
            f"The connected AI model returned {label} "
            f"with {confidence}% confidence."
        ),
        "patientExplanation": (
            "This is an AI-based prototype screening result "
            "and is not a medical diagnosis."
        ),
        "recommendation": (
            "Please have the result reviewed by a "
            "qualified eye-care professional."
        ),
        "findings": [],
        "visualization": result["visualization"]
    }