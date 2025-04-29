import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import streamlit as st
import urllib.request
import json

# Load class labels
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
with urllib.request.urlopen(LABELS_URL) as url:
    labels = json.load(url)

# Load Pretrained Model (ResNet-50)
model = models.resnet50(pretrained=True)
model.eval()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("🐶 Dog Breed Classifier")

uploaded_file = st.file_uploader("Upload an image of a dog..", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess and predict
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
    
    # Get predicted class
    _, predicted_idx = outputs.max(1)
    predicted_label = labels[str(predicted_idx.item())][1]
    
    st.write(f"### Predicted Breed: **{predicted_label.replace('_', ' ')}** 🐕")
