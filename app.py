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
# Custom UI Styling
# --------------------------
st.markdown("""
<style>
.big-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #00bcd4;
}
.sub-text {
    text-align: center;
    color: #aaaaaa;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Header
# --------------------------
st.markdown('<p class="big-title">🌊 Microplastic Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">AI-powered water quality analysis using MobileNetV2</p>', unsafe_allow_html=True)

st.markdown("---")

# --------------------------
# Load TFLite Model
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
st.sidebar.title("📌 About Project")
st.sidebar.write("""
This system uses Deep Learning (MobileNetV2)  
to detect microplastics in water samples.
""")

threshold = st.sidebar.slider("Detection Threshold", 0.3, 0.9, 0.5)

# --------------------------
# Upload Section
# --------------------------
st.markdown("## 📤 Upload Water Sample Image")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    # Show Image
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # --------------------------
    # Preprocessing (IMPORTANT)
    # --------------------------
    img = image.resize((224, 224))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # --------------------------
    # Prediction
    # --------------------------
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    confidence = float(prediction)

    # --------------------------
    # Result UI
    # --------------------------
    with col2:
        st.markdown("## 🔍 Prediction Result")

        if confidence >= threshold:
            st.error("⚠️ Microplastic Detected")
        else:
            st.success("💧 Clean Water")

        st.metric("Confidence Score", f"{confidence*100:.2f}%")

        # Low confidence warning
        if 0.4 < confidence < 0.6:
            st.warning("⚠️ Model is uncertain about this prediction")

        # Progress bar
        st.progress(confidence)

        # --------------------------
        # Graph
        # --------------------------
        st.markdown("### 📊 Confidence Distribution")

        labels = ['Clean Water', 'Microplastic']
        values = [1-confidence, confidence]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_ylim(0,1)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")

        st.pyplot(fig)

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown("👨‍💻 Final Year Project | Microplastic Detection using Deep Learning")
