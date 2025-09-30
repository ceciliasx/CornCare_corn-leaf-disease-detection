import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import base64

st.set_page_config(page_title="Detect Corn Leaf Disease", layout="centered")

main_bg = "corn_leaf_bg.jpg"
main_bg_ext = "jpg"

with open(main_bg, "rb") as file:
    base64_jpg = base64.b64encode(file.read()).decode()

# CSS: transparent overlay
st.markdown(f"""
<style>
[data-testid="stApp"] {{
    background: 
        linear-gradient(rgba(255,255,255,0.3), rgba(255,255,255,0.3)), 
        url("data:image/{main_bg_ext};base64,{base64_jpg}");
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center;
}}

/* Dark mode */
[data-testid="stApp"][data-dark] {{
    background: 
        linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.3)), 
        url("data:image/{main_bg_ext};base64,{base64_jpg}");
}}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource(show_spinner=True)
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes
    model.load_state_dict(torch.load("best_ResNet.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()
class_names = ["hawar", "karat", "sehat"]

# Transform image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# UI
st.title("🌽 Corn Leaf Disease Detector")
st.markdown("### Detect Hawar (Blight), Karat (Rust), or Sehat (Healthy) Corn Leaves")
st.text("Model: ResNet50 | Dataset: 3000+ Images")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="*Uploaded Image*", use_container_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        label = class_names[pred.item()]

    st.markdown(f"### ╰┈➤  Prediction: **{label}**")
    if label == "sehat":
        st.balloons()