import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model("savedmodel")
class_names = ["Early Blight", "Late Blight", "Healthy"]

# Function to capture image from the camera
def capture_image():
    ret, frame = cap.read()
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Streamlit UI
st.title("Potato Leaf Disease Model")

# Option to choose between file upload and camera capture
option = st.radio("Choose Input Method:", ("File Upload", "Camera Capture"))

if option == "File Upload":
    uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "jpeg", "png"], key="file_uploader")
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        # Image preprocessing
        image_np = np.array(image)
        img_batch = np.expand_dims(image_np, 0)

        # Make predictions
        with st.spinner("Making Predictions..."):
            predictions = model.predict(img_batch)

        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        # Display prediction result
        st.subheader("Prediction Result:")
        st.write(f"Class: {predicted_class}")

        # Add color or icon based on the predicted class
        if predicted_class == "Early Blight":
            st.success("Early Blight Detected")
        elif predicted_class == "Late Blight":
            st.warning("Late Blight Detected")
        elif predicted_class == "Healthy":
            st.info("Healthy Potato Leaf")

        # Display confidence
        st.subheader("Prediction Confidence:")
        confidence_percentage = confidence * 100
        st.progress(int(confidence_percentage))
        st.write(f"Confidence: {confidence_percentage:.2f}%")

else:
    # Capture image from the camera using OpenCV
    cap = cv2.VideoCapture(0)

    # Display placeholders for the live camera feed and the captured frame
    camera_placeholder = st.empty()
    captured_frame_placeholder = st.empty()

    # Button to capture a single frame
    if st.button("Capture Frame"):
        captured_image = capture_image()

        # Resize captured image to match model input shape
        resized_image = captured_image.resize((256, 256))

        # Display the captured frame
        captured_frame_placeholder.image(resized_image, caption="Captured Frame", use_column_width=True)

        # Image preprocessing
        image_np = np.array(resized_image)
        img_batch = np.expand_dims(image_np, 0)

        # Make predictions
        with st.spinner("Making Predictions..."):
            predictions = model.predict(img_batch)

        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        # Display prediction result
        st.subheader("Prediction Result:")
        st.write(f"Class: {predicted_class}")

        # Add color or icon based on the predicted class
        if predicted_class == "Early Blight":
            st.success("Early Blight Detected")
        elif predicted_class == "Late Blight":
            st.warning("Late Blight Detected")
        elif predicted_class == "Healthy":
            st.info("Healthy Potato Leaf")

        # Display confidence
        st.subheader("Prediction Confidence:")
        confidence_percentage = confidence * 100
        st.progress(int(confidence_percentage))
        st.write(f"Confidence: {confidence_percentage:.2f}%")

    # Checkbox to show/hide live camera feed
    show_live_feed = st.checkbox("Show Live Camera Feed", key="show_live_feed_checkbox")

    # Continuously display the live camera feed
    while show_live_feed:
        captured_image = capture_image()

        # Resize captured image to match model input shape
        resized_image = captured_image.resize((256, 256))

        # Display the live camera feed
        camera_placeholder.image(resized_image, caption="Live Camera Feed", use_column_width=True)

    # Release the camera when the checkbox is unchecked
    cap.release()
