import os
# 🔥 FIX protobuf issue (VERY IMPORTANT)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(page_title="Microplastic Detection", layout="wide")

st.title("🌊 Microplastic Detection System")
st.markdown("AI-powered water quality analysis")

# --------------------------
# LOAD TFLITE MODEL
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
# UPLOAD IMAGE
# --------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # --------------------------
    # AUTO PREPROCESS (SAFE)
    # --------------------------
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    height = input_shape[1]
    width = input_shape[2]

    img = image.resize((width, height))
    img = np.array(img)

    img = np.expand_dims(img, axis=0)

    if input_dtype == np.float32:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.uint8)

    # --------------------------
    # PREDICTION
    # --------------------------
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    confidence = float(prediction)

    # --------------------------
    # RESULT (SAFE LOGIC)
    # --------------------------
    st.write("Raw Output:", confidence)

    if confidence > 0.5:
        label = "Microplastic"
        score = confidence
    else:
        label = "Clean Water"
        score = 1 - confidence

    with col2:
        st.subheader("🔍 Prediction Result")

        if label == "Microplastic":
            st.error(f"⚠️ Microplastic Detected ({score*100:.2f}%)")
        else:
            st.success(f"💧 Clean Water ({score*100:.2f}%)")

        st.progress(score)

        # --------------------------
        # GRAPH
        # --------------------------
        st.markdown("### 📊 Confidence")

        clean_prob = 1 - confidence
        micro_prob = confidence

        fig, ax = plt.subplots()
        ax.bar(["Clean", "Microplastic"], [clean_prob, micro_prob])
        ax.set_ylim(0, 1)

        st.pyplot(fig)

# --------------------------
# FOOTER
# --------------------------
st.markdown("---")
st.markdown("Final Year Project | Microplastic Detection")
