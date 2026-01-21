import streamlit as st
import numpy as np
import pickle
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Vehicle Classifier",
    page_icon="🚘",
    layout="centered"
)

# ---------------- BACKGROUND IMAGE (URL) ----------------
BG_IMAGE_URL = "https://www.shutterstock.com/image-photo/outdoor-photo-car-motorcycle-driving-600nw-2679840843.jpg"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{BG_IMAGE_URL}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .glass {{
        background: rgba(0, 0, 0, 0.55);
        border-radius: 18px;
        padding: 35px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }}

    h1, h3, p, label {{
        color: #ffffff !important;
    }}

    .stButton>button {{
        background: linear-gradient(135deg, #ff4b2b, #ff416c);
        color: white;
        font-weight: bold;
        border-radius: 12px;
        height: 3em;
        width: 100%;
    }}

    .stFileUploader label {{
        color: white !important;
        font-weight: 600;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("vehicle_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_map.pkl", "rb") as f:
        label_map = pickle.load(f)
    return model, label_map

model, label_map = load_model()

# ---------------- UI CARD ----------------
st.markdown(
    """
    <div class="glass">
        <h1 style="text-align:center;">🚗 Vehicle Classification</h1>
        <p style="text-align:center; font-size:18px;">
        Upload an image to classify it as <b>Bike</b> or <b>Car</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

uploaded_file = st.file_uploader(
    "📤 Upload Vehicle Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION ----------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, -1)

    with st.spinner("🔍 Classifying vehicle..."):
        pred = model.predict(img_array)[0]
        result = label_map[pred]

    st.markdown(
        f"""
        <div class="glass">
            <h3 style="text-align:center;">Prediction Result</h3>
            <h1 style="text-align:center;">✅ {result.upper()}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- FOOTER ----------------
st.markdown(
    "<p style='text-align:center; color:white;'>Powered by Machine Learning • Streamlit • Hugging Face</p>",
    unsafe_allow_html=True
)
