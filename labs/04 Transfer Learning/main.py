from fastapi import FastAPI
import torch
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io

app = FastAPI()

# TODO load the pre-trained model
model = None


# Example inference
def predict(image, model):
    # TODO implement prediction to return string value of predicted class
    return "Apple"


@app.post("/predict/")
def upload_image(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    image = Image.open(io.BytesIO(image_bytes))
    predicted_class = predict(image, model)
    return predicted_class
