import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json

st.title("Traffic Sign Classification")

img_size = 64
model = tf.keras.models.load_model("traffic_sign_model.h5")

with open("classes.json", "r") as f:
    label_map = json.load(f)

uploaded_file = st.file_uploader("Upload a traffic sign image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR", caption="Uploaded Image")
    img_resized = cv2.resize(img, (img_size, img_size))
    img_resized = np.expand_dims(img_resized, axis=0) / 255.0
    prediction = model.predict(img_resized)
    class_id = int(np.argmax(prediction))
    class_name = label_map[str(class_id)]
    st.success(f"Prediction: {class_name}")

