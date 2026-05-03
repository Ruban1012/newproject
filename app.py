import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="Microplastic Detection", layout="wide")

# --------------------------
# UI
# --------------------------
st.title("🌊 Microplastic Detection System")
st.markdown("AI-powered water quality analysis")

st.markdown("---")

# --------------------------
# Load Model
# --------------------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --------------------------
# Sidebar
# --------------------------
threshold = st.sidebar.slider("Detection Threshold", 0.3, 0.9, 0.5)

# --------------------------
# Upload
# --------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # --------------------------
    # AUTO PREPROCESS
    # --------------------------
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    height = input_shape[1]
    width = input_shape[2]

    img = image.resize((width, height))
    img = np.array(img)

    # Fix RGBA → RGB
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img = np.expand_dims(img, axis=0)

    # Match dtype
    if input_dtype == np.float32:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.uint8)

    # --------------------------
    # Prediction
    # --------------------------
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    confidence = float(prediction)

    # --------------------------
    # CORRECT LABEL LOGIC
    # --------------------------
    if confidence >= 0.5:
        label = "Microplastic"
    else:
        label = "Clean Water"

    # --------------------------
    # UI RESULT
    # --------------------------
    with col2:
        st.subheader("🔍 Prediction Result")

        st.write("Raw Prediction:", confidence)  # DEBUG (remove later)

        if label == "Microplastic":
            st.error(f"⚠️ Microplastic Detected ({confidence*100:.2f}%)")
        else:
            st.success(f"💧 Clean Water ({(1-confidence)*100:.2f}%)")

        # Progress bar
        st.progress(confidence)

        # --------------------------
        # Graph
        # --------------------------
        st.markdown("### 📊 Confidence")

        labels = ["Clean", "Microplastic"]
        values = [1-confidence, confidence]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_ylim(0, 1)

        st.pyplot(fig)

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown("Final Year Project | Microplastic Detection")
