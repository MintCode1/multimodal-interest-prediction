import streamlit as st
import requests

st.title("Multimodal Engagement Predictor Dashboard")

# Number input to let user pick video index
index = st.number_input("Select Video Index", min_value=0, max_value=300, value=0, step=1)

# Button to trigger prediction
if st.button("Predict Engagement"):
    # Send POST request to FastAPI backend
    response = requests.post("http://127.0.0.1:8000/predict/", json={"index": index})
    
    if response.status_code == 200:
        prob = response.json()["engagement_probability"]
        st.metric(label="Predicted Engagement Probability", value=f"{prob:.2f}")
    else:
        st.error("Prediction request failed! Check if API is running and index is valid.")

# Info text for user
st.info("Use the number input above to choose which video's features to predict on. "
        "Update your preprocessed feature files if you add more videos!")
