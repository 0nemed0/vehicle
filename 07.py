import streamlit as st
import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 model (assuming you have it saved or using a pre-trained one)
model = YOLO('yolov8n.pt')  # You can use other YOLOv8 models like yolov8m.pt, etc.

# Define options for video input
st.title("Vehicle Detection with YOLOv8")

option = st.selectbox(
    'Select Input Type:',
    ('Recorded Video', 'Live Webcam', 'External Camera')
)

def detect_vehicles(frame):
    # Perform vehicle detection using YOLOv8 model
    results = model(frame)
    return results

if option == 'Recorded Video':
    # Upload a recorded video file
    video_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    
    if video_file is not None:
        # Convert the uploaded file to a video stream
        video = cv2.VideoCapture(video_file.name)
        stframe = st.empty()

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Detect vehicles
            results = detect_vehicles(frame)
            
            # Display results
            annotated_frame = results.render()[0]
            stframe.image(annotated_frame, channels="BGR")

        video.release()

elif option == 'Live Webcam':
    # Access live webcam feed (ID=0 for built-in webcam)
    video = cv2.VideoCapture(0)
    stframe = st.empty()

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            st.warning("Webcam feed not available.")
            break

        # Detect vehicles
        results = detect_vehicles(frame)
        
        # Display results
        annotated_frame = results.render()[0]
        stframe.image(annotated_frame, channels="BGR")

    video.release()

elif option == 'External Camera':
    # Access external camera feed (usually ID=1 or a different IP stream)
    camera_url = st.text_input("Enter external camera stream URL or ID:", "http://your_ip_camera_feed")
    
    if camera_url:
        video = cv2.VideoCapture(camera_url)
        stframe = st.empty()

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                st.warning("External camera feed not available.")
                break

            # Detect vehicles
            results = detect_vehicles(frame)

            # Display results
            annotated_frame = results.render()[0]
            stframe.image(annotated_frame, channels="BGR")

        video.release()

# Release resources after stop
cv2.destroyAllWindows()
