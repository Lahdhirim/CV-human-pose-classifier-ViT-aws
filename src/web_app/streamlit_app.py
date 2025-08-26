import streamlit as st
import requests
from PIL import Image
import matplotlib.pyplot as plt

API_URL = "http://localhost:8502/api/v1/pose_classifier"

st.title("Human Pose Classification")

# Input URL
image_url = st.text_input("Enter the image URL:")

if st.button("Predict") and image_url:
    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
        img = Image.open(requests.get(image_url, stream=True).raw)
        st.image(img, caption="Input Image", use_column_width=True)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()

    try:
        response = requests.post(API_URL, json={"url": image_url})
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        st.error(f"Error calling API: {e}")
        st.stop()

    top_labels = result.get("top_labels", [])
    top_probs = result.get("top_probs", [])

    if top_labels and top_probs:
        st.subheader("Top Predictions")
        fig, ax = plt.subplots()
        ax.pie(top_probs, labels=top_labels, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)
    else:
        st.warning("No predictions returned.")
