import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Microplastic Detection", layout="centered")

# =========================
# DARK THEME UI
# =========================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
}
.big-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #00d4ff;
}
.sub-text {
    text-align: center;
    color: #aaaaaa;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown('<p class="big-title">🌊 Microplastic Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">AI-powered water quality analysis</p>', unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("About Project")
st.sidebar.write("""
This system uses Deep Learning (MobileNetV2) 
to detect microplastics in water samples.
""")

# =========================
# BUILD MODEL
# =========================
def build_model():
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224,224,3))

    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = build_model()
    model.load_weights("model.weights.h5")
    return model

model = load_model()

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224,224))

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # =========================
    # PREPROCESS
    # =========================
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # =========================
    # PREDICTION
    # =========================
    prediction = model.predict(img_array)
    confidence = prediction[0][0]

    # =========================
    # RESULT
    # =========================
    st.subheader("🔍 Prediction Result")

    if confidence > 0.5:
        st.error("🧪 Microplastic Detected")
        st.metric("Confidence", f"{confidence*100:.2f}%")
    else:
        st.success("💧 Clean Water")
        st.metric("Confidence", f"{(1-confidence)*100:.2f}%")

    # =========================
    # PROBABILITIES
    # =========================
    clean_prob = float(1 - confidence)
    micro_prob = float(confidence)

    # =========================
    # 📊 BAR GRAPH (FIXED)
    # =========================
    st.subheader("📊 Confidence Bar Graph")

    df = pd.DataFrame({
        "Class": ["Clean Water", "Microplastic"],
        "Probability": [clean_prob, micro_prob]
    })

    st.bar_chart(df.set_index("Class"))

    # =========================
    # 🍩 DONUT CHART
    # =========================
    st.subheader("🍩 Prediction Distribution")

    fig = go.Figure(data=[go.Pie(
        labels=['Clean Water', 'Microplastic'],
        values=[clean_prob, micro_prob],
        hole=0.6
    )])

    fig.update_traces(textinfo='percent+label', pull=[0.05, 0.05])
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # 🎯 GAUGE FIXED
    # =========================
    st.subheader("🎯 Confidence Meter")

    if confidence > 0.5:
        gauge_value = confidence * 100
    else:
        gauge_value = (1 - confidence) * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_value,
        title={'text': "Confidence (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "cyan"},
            'steps': [
                {'range': [0, 50], 'color': "#ff4b4b"},
                {'range': [50, 80], 'color': "#ffa600"},
                {'range': [80, 100], 'color': "#00ff9c"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # 📊 PROGRESS BAR
    # =========================
    st.subheader("📊 Confidence Level")
    st.progress(int(gauge_value))