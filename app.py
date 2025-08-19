import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

model = tf.keras.models.load_model("traffic_model.h5")


meta = pd.read_csv("meta.csv")
class_map = dict(zip(meta["ClassId"], meta["SignName"]))

st.title("Traffic Sign Recognition App")

uploaded_file = st.file_uploader("Upload a traffic sign image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((30, 30))  
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    predictions = model.predict(img_array)
    predicted_class_id = np.argmax(predictions)

    predicted_class_name = class_map.get(predicted_class_id, "Unknown Sign")

    st.write(f"**Predicted Sign:** {predicted_class_name} (ID: {predicted_class_id})")

