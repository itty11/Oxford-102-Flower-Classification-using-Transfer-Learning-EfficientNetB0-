import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import os
import matplotlib.pyplot as plt

# CONFIG

st.set_page_config(page_title="🌸 Flower Classification App", layout="centered")

MODEL_PATH = "models/flower_classifier.keras"
CLASS_MAP_PATH = "cat_to_name.json"  # mapping from class indices to flower names


# LOAD MODEL

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()


# LOAD CLASS LABELS

if os.path.exists(CLASS_MAP_PATH):
    with open(CLASS_MAP_PATH, "r") as f:
        class_names = json.load(f)
    st.sidebar.success("✅ Loaded class name mapping from JSON.")
else:
    st.sidebar.warning("⚠️ 'cat_to_name.json' not found. Using class indices as labels.")
    # fallback — load labels from training generator indices if available
    num_classes = model.layers[-1].output_shape[-1]
    class_names = {str(i): f"Class {i}" for i in range(num_classes)}

# Convert class_names to sorted list
sorted_classes = [class_names[k] for k in sorted(class_names.keys(), key=lambda x: int(x))]


# IMAGE PREPROCESSING

def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img, img_array


# APP UI

st.title("🌸 Oxford 102 Flower Classification")
st.write("Upload a flower image and let the model predict its type!")

uploaded_file = st.file_uploader("📤 Upload a flower image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img, img_array = preprocess_image(uploaded_file)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Predict
    preds = model.predict(img_array)
    predicted_idx = np.argmax(preds)
    predicted_label = sorted_classes[predicted_idx]
    confidence = preds[0][predicted_idx] * 100

    st.markdown(f"### 🌼 Predicted: **{predicted_label}**")
    st.markdown(f"#### 🔹 Confidence: **{confidence:.2f}%**")

    # Show top-5 probabilities
    top_indices = preds[0].argsort()[-5:][::-1]
    top_confidences = preds[0][top_indices] * 100
    top_labels = [sorted_classes[i] for i in top_indices]

    st.write("#### 📊 Top 5 Predictions")
    fig, ax = plt.subplots()
    ax.barh(top_labels[::-1], top_confidences[::-1], color="skyblue")
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Top 5 Predictions")
    st.pyplot(fig)

else:
    st.info("👆 Upload an image to start classification.")


# FOOTER

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + TensorFlow | Oxford 102 Flower Dataset")
