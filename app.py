import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import base64

st.set_page_config(page_title="CornCare: Detect Corn Leaf Disease", layout="centered")

main_bg = "corn_leaf_bg.jpg"
with open(main_bg, "rb") as file:
    base64_jpg = base64.b64encode(file.read()).decode()

st.markdown(f"""
<style>
[data-testid="stApp"] {{
    background:
        linear-gradient(rgba(255,255,255,0.3), rgba(255,255,255,0.3)),
        url("data:image/jpg;base64,{base64_jpg}");
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center;
}}
[data-testid="stApp"][data-dark] {{
    background:
        linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.3)),
        url("data:image/jpg;base64,{base64_jpg}");
}}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=True)
def load_model(model_name):
    num_classes = 3

    if model_name == "ResNet50":
        model = models.resnet50(pretrained=False)

        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        model.load_state_dict(
            torch.load("best_ResNet.pth", map_location="cpu")
        )

    elif model_name == "EfficientNet-B0":
        model = models.efficientnet_b0(pretrained=False)

        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        model.load_state_dict(
            torch.load("best_EfficientNet.pth", map_location="cpu")
        )

    else:
        raise ValueError("Model not supported")

    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

class_names = ["hawar", "karat", "sehat"]

st.title("🌽 CornCare")
st.subheader("Corn Leaf Disease Detection")
st.markdown("### Detect Hawar (Blight), Karat (Rust), or Sehat (Healthy)")
st.text("Dataset: 3000+ Images")

# 🔽 Model selection
model_choice = st.radio(
    "Select Model",
    ["ResNet50", "EfficientNet-B0"],
    horizontal=True
)

model = load_model(model_choice)

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    label = class_names[pred]
    confidence = probs[0][pred].item() * 100

    st.markdown(
        f"### ╰┈➤ Prediction: **{label.upper()}**  \n"
        f"**Confidence:** {confidence:.2f}%"
    )

    if label == "sehat":
        st.balloons()