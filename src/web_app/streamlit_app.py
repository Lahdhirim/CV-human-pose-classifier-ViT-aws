import streamlit as st
import requests
from PIL import Image
import matplotlib.pyplot as plt

# URL of FastAPI endpoint
API_URL = "http://localhost:8502/api/v1/pose_classifier"

st.set_page_config(page_title="Human Pose Classification", page_icon="🧍")
st.title("🧍Human Pose Classification App")

# Initialize session state for processing status
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# Input URL
image_url = st.text_input("Enter the image URL:")

# Predict Button is disabled if no URL or if currently processing
button_disabled = not image_url

if st.button("Predict", disabled=button_disabled):

    with st.spinner("🔄 Processing in progress... Please wait."):

        # Load image preview
        try:
            response = requests.get(image_url, timeout=5)
            response.raise_for_status()
            img = Image.open(requests.get(image_url, stream=True).raw)
            st.image(img, caption="Input Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            st.stop()

        # Call FastAPI for prediction
        try:
            response = requests.post(API_URL, json={"url": image_url})
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            st.error(f"Error calling API: {e}")
            st.stop()

        # Extract predictions
        top_labels = result.get("top_labels", [])
        top_probs = result.get("top_probs", [])
        prediction_time = result.get("prediction_time", None)

        if top_labels and top_probs:
            st.success("✅ Prediction completed successfully.")

            # Histogram of top probabilities
            st.subheader("Top Predictions")
            fig, ax = plt.subplots()
            ax.bar(top_labels, top_probs)
            ax.set_xlabel("Probability")
            ax.set_ylabel("Class")
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

            # Display calculation time
            if prediction_time:
                st.caption(f"⏱ Prediction time: {prediction_time} ms")
        else:
            st.warning("⚠️ No predictions returned.")
