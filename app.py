import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes
    model.load_state_dict(torch.load("corn_resnet50.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()
class_names = ["hawar", "karat", "sehat"]

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("🌽 Corn Leaf Disease Detector (ResNet50)")
st.write("Upload a corn leaf image to detect whether it is **Hawar, Karat, or Sehat (Healthy)**")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        label = class_names[pred.item()]

    st.markdown(f"### 🟢 Prediction: **{label}**")