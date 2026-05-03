import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tflite_runtime.interpreter import Interpreter

st.set_page_config(page_title="Microplastic Detection", layout="wide")

st.title("🌊 Microplastic Detection System")

# --------------------------
# Load TFLite Model
# --------------------------
@st.cache_resource
def load_model():
    interpreter = Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --------------------------
# Upload
# --------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    # --------------------------
    # AUTO PREPROCESS
    # --------------------------
    shape = input_details[0]['shape']
    dtype = input_details[0]['dtype']

    h, w = shape[1], shape[2]

    img = image.resize((w, h))
    img = np.array(img)

    img = np.expand_dims(img, axis=0)

    if dtype == np.float32:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.uint8)

    # --------------------------
    # Predict
    # --------------------------
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

    st.write("Raw:", float(pred))

    if pred > 0.5:
        st.error("⚠️ Microplastic")
    else:
        st.success("💧 Clean Water")

    st.progress(float(pred))
