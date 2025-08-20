import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd
from tensorflow.keras.utils import to_categorical

st.title("Traffic Sign Classification")

img_size = 64
X, y = [], []
train_df = pd.read_csv("archive/Train.csv")
meta = pd.read_csv("archive/meta.csv")

for i in range(2000):
    img_path = os.path.join("archive", train_df['Path'][i])
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.resize(img, (img_size, img_size))
    X.append(img)
    y.append(train_df['ClassId'][i])

X = np.array(X) / 255.0
y = to_categorical(y, num_classes=len(meta))

model = tf.keras.models.load_model("traffic_sign_model.h5")

uploaded_file = st.file_uploader("Upload a traffic sign image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR", caption="Uploaded Image")
    img_resized = cv2.resize(img, (img_size, img_size))
    img_resized = np.expand_dims(img_resized, axis=0) / 255.0
    prediction = model.predict(img_resized)
    class_id = np.argmax(prediction)
    class_name = meta.loc[meta['ClassId'] == class_id, 'SignName'].values[0]
    st.success(f"Prediction: {class_name}")
