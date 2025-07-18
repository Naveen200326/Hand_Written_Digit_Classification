import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

st.set_page_config(page_title="Digit Classifier", page_icon="🧠")
st.title("📷 Upload a Handwritten Digit")
st.caption("Upload an image of a digit (0–9), and the model will predict it using a CNN trained on MNIST.")

# Load model
model = load_model("mnist_model.h5")

# File uploader
uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preview image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded Image", width=200)

    # Preprocess image
    img = np.array(image)

    # Resize to 28x28
    img = cv2.resize(img, (28, 28))

    # Invert colors: black background, white digit
    img = 255 - img

    # Normalize and reshape
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img)
    predicted_digit = int(np.argmax(prediction))

    st.subheader(f"🧠 Predicted Digit: `{predicted_digit}`")
else:
    st.info("📤 Please upload an image of a handwritten digit.")
