# app.py
import streamlit as st
import torch
from PIL import Image
import numpy as np
import altair as alt
import pandas as pd

from model_def import LeNet4, IMG_SIZE
from inference import predict, CLASS_NAMES

st.set_page_config(page_title="Cervical Cytology Classifier", page_icon="🧫", layout="centered")

# Sidebar: device + weights
st.sidebar.title("Settings")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write(f"Device: {device}")

weights_path = "weights/cervical_lenet4.pth"
st.sidebar.write(f"Model weights: {weights_path}")

# Load model once
# ...existing code...
@st.cache_resource
def load_model(weights_path, device):
    model = LeNet4(num_classes=len(CLASS_NAMES)).to(device)
    try:
        # Attempt to load weights
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model
    except EOFError:
        st.error(f"Weights file appears corrupted or truncated: {weights_path}")
        st.warning("Using uninitialized model. Re-download or re-generate the weights file and restart the app.")
        return model  # empty model so app keeps running
    except FileNotFoundError:
        st.error(f"Weights file not found: {weights_path}")
        st.info("Place the weights file at the path above or update weights_path in the sidebar.")
        return model
    except RuntimeError as e:
        st.error(f"Runtime error loading state_dict: {e}")
        st.warning("Model architecture and weights may mismatch.")
        return model
    except Exception as e:
        st.error(f"Unexpected error loading weights: {e}")
        return model

model = load_model(weights_path, device)

st.title("Cervical Cytology Classification")
st.caption("Mendeley LBC, LeNet-style CNN (IMG_SIZE=128), 4 classes")

uploaded = st.file_uploader("Upload a cervical cytology image", type=["png", "jpg", "jpeg", "bmp", "tiff"])

if uploaded is not None:
    pil_image = Image.open(uploaded)
    st.image(pil_image, caption="Input image", use_column_width=True)

    logits, probs, pred_name = predict(model, device, pil_image)

    st.subheader(f"Prediction: {pred_name}")
    st.write("Confidence per class:")

    # Bar chart for probabilities
    data = pd.DataFrame({
        "class": CLASS_NAMES,
        "prob": probs.tolist()
    })

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("class:N", sort=None, title="Class"),
            y=alt.Y("prob:Q", title="Probability"),
            tooltip=["class", "prob"]
        )
        .properties(width=600, height=400)
    )

    st.altair_chart(chart, use_container_width=True)

    # Raw logits if needed
    with st.expander("Show raw logits"):
        st.write({CLASS_NAMES[i]: float(logits[i]) for i in range(len(CLASS_NAMES))})

else:
    st.info("Upload a cytology image to get started.")

st.markdown("---")
st.markdown("Notes:")
st.markdown("- The model architecture and normalization match the training notebook.")
st.markdown("- Predictions are based on resized 128×128 inputs with ImageNet normalization.")