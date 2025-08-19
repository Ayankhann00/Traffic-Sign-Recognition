import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd

@st.cache_resource
def load_trained_model():
    return load_model("traffic_sign_model.h5")

model = load_trained_model()


st.title("🚦 Traffic Sign Recognition")
st.write("Upload a traffic sign image, and the model will predict its class.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

img_size = 64

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (img_size, img_size)) / 255.0
    img_exp = np.expand_dims(img_resized, axis=0)

    prediction = model.predict(img_exp)
    pred_class = np.argmax(prediction)

    st.write(f"Predicted Class ID: {pred_class}")

    if "ClassId" in meta.columns and "SignName" in meta.columns:
        if pred_class in meta["ClassId"].values:
            sign_name = meta[meta["ClassId"] == pred_class]["SignName"].values[0]
            st.success(f"🚦 Predicted Traffic Sign: {sign_name}")

