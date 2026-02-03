import streamlit as st
import cv2
from src.driver_drowsiness_project.pipelines.prediction_pipeline import DrowsinessPredictor


model_path = "artifacts/model/driver_drowsiness_model.keras"

st.set_page_config(page_title="Driver Drowsiness Detection",layout="wide")
st.title("🚗 Driver Drowsiness Detection (Live Webcam)")

predictor = DrowsinessPredictor(model_path)

run = st.checkbox("▶ Start Webcam")

frame_window = st.image([])

cap = None

if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret,frame = cap.read()
        if not ret:
            st.error("Failed to access Webcam")

            break

        label,confidence =  predictor.predict(frame)

        # Draw Prediction

        color = (0,255,0)

        if label in ["Closed","no_yawn"]:
            color = (0,0,255)

        cv2.putText(
            frame,
            f"{label} ({confidence:.2f})",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame)
else:
    if cap:
        cap.release()



