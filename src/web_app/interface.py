import streamlit as st
import requests
from PIL import Image
import matplotlib.pyplot as plt

# URL of FastAPI endpoint
API_URL = "http://localhost:8502/api/v1/pose_classifier"

st.set_page_config(page_title="Human Pose Classification", page_icon="🧍")
st.title("🧍Human Pose Classification App")

# --- Choose input method ---
input_method = st.radio("Choose input method:", ["Image URL", "Upload File"])

image_url = None
uploaded_file = None

if input_method == "Image URL":
    image_url = st.text_input("Enter the image URL:")
elif input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Disable predict button if nothing is provided
button_disabled = (input_method == "Image URL" and not image_url) or (
    input_method == "Upload File" and not uploaded_file
)

if st.button("Predict", disabled=button_disabled):

    with st.spinner("🔄 Processing in progress... Please wait."):

        # Load image preview
        try:
            if image_url:
                response = requests.get(image_url, timeout=5)
                response.raise_for_status()
                img = Image.open(requests.get(image_url, stream=True).raw)
            else:
                img = Image.open(uploaded_file)
            st.image(img, caption="Input Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            st.stop()

        # Call FastAPI for prediction
        files = {"file": uploaded_file.getvalue()} if uploaded_file else None
        data = {"url": image_url} if image_url else None
        try:
            response = requests.post(API_URL, data=data, files=files)
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
