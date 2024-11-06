# main.py
from fastapi import FastAPI, File, UploadFile
import torch
from model import load_model, preprocess_image
from PIL import Image
import io

app = FastAPI()

# Load the trained model
model = load_model("models/fine_tuned_marine_animal_classifier.pth")  # Update the path if necessary

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    # Load and preprocess the image
    image = Image.open(io.BytesIO(await file.read()))
    image_tensor = preprocess_image(image)
    # Run the model
    with torch.no_grad():
        outputs = model(image_tensor)
    _, predicted = outputs.max(1)
    return {"class": predicted.item()}
