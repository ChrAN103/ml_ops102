import os

import pandas as pd
import requests
import streamlit as st

import traceback

# def get_backend_url():
#     """Get the URL of the backend service."""
#     parent = "projects/my-personal-mlops-project/locations/europe-west1"
#     client = run_v2.ServicesClient()
#     services = client.list_services(parent=parent)
#     for service in services:
#         if service.name.split("/")[-1] == "production-model":
#             return service.uri
#     return os.environ.get("BACKEND", None)


def classify_news(title, text):
    """Send the article to the backend for classification."""
    predict_url = f"https://my-fastapi-service-358298092496.europe-west1.run.app/predict"
    response = requests.post(predict_url, json={"title": title, "text": text}, timeout=10)
    if response.status_code == 200:
        return response.json()
    return response.text


def main() -> None:
    """Main function of the Streamlit frontend."""
    st.title("Fake News Classification")

    title = st.text_input("Enter article title")
    text = st.text_area("Enter article text")

    if st.button("Classify"):
        result = classify_news(title, text)

        try:
            prediction = result["prediction"]
            probability = result["prob"]
            
            # show the prediction and probability for the highest class
            predicted_class = "Real" if prediction else "Fake News"
            st.write("Prediction:", predicted_class)
            st.write("Probability:", probability)
        except Exception as e:
            st.write(f"Failed to get prediction. Response: {result}. error: {e}. Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()