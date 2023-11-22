import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model("savedmodel")
class_names = ["Early Blight", "Late Blight", "Healthy"]

# Streamlit UI
st.title("Potato Leaf Disease Model")

# Option to choose between file upload and camera capture
option = st.radio("Choose Input Method:", ("File Upload", "Camera Capture"))

if option == "File Upload":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
else:
    # Capture image from the camera using OpenCV
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

# Display the captured image
if image is not None:
    st.image(image, caption="Captured Image.", use_column_width=True)

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
