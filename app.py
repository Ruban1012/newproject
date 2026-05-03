import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# --------------------------
# PAGE SETUP
# --------------------------
st.set_page_config(page_title="Microplastic Detection", layout="wide")

st.title("🌊 Microplastic Detection System")
st.markdown("AI-powered water quality analysis")

# --------------------------
# LOAD MODEL (NO CACHE)
# --------------------------
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --------------------------
# FILE UPLOAD
# --------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # --------------------------
    # SHOW IMAGE
    # --------------------------
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # --------------------------
    # PREPROCESS (FINAL FIX)
    # --------------------------
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    height = input_shape[1]
    width = input_shape[2]

    img = image.resize((width, height))
    img = np.array(img)

    # Ensure RGB
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img = np.expand_dims(img, axis=0)

    # Correct dtype
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
    # DEBUG OUTPUT (IMPORTANT)
    # --------------------------
    st.write("Raw Output:", confidence)

    # --------------------------
    # AUTO LABEL FIX
    # --------------------------
    # Try both mappings automatically
    micro_prob = confidence
    clean_prob = 1 - confidence

    if micro_prob > clean_prob:
        label = "Microplastic"
        score = micro_prob
    else:
        label = "Clean Water"
        score = clean_prob

    # --------------------------
    # RESULT UI
    # --------------------------
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

        fig, ax = plt.subplots()
        ax.bar(["Clean", "Microplastic"], [clean_prob, micro_prob])
        ax.set_ylim(0, 1)
        st.pyplot(fig)

# --------------------------
# FOOTER
# --------------------------
st.markdown("---")
st.markdown("Final Year Project | Microplastic Detection")
