import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model("savedmodel")
class_names = ["Early Blight", "Late Blight", "Healthy"]

# Streamlit UI
st.title("Potato Leaf Disease Model")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Image preprocessing
    image_np = np.array(image)
    img_batch = np.expand_dims(image_np, 0)

    # Make predictions
    predictions = model.predict(img_batch)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    # Display prediction result
    st.subheader("Prediction Result:")
    st.write(f"Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2%}")
