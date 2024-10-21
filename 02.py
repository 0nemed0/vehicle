import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from sort import Sort
from PIL import Image
import tempfile

# Load YOLOv8 model (pre-trained)
model = YOLO('yolov8n.pt')

# Vehicle class IDs from the COCO dataset
vehicle_classes = [2, 3, 5, 7]  # 2: car, 3: motorcycle, 5: bus, 7: truck
class_names = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}

# Initialize the SORT tracker
tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

# Streamlit app title and instructions
st.title('Real-Time Vehicle Detection and Tracking')
st.write("Upload a video file to detect, track, and count vehicles in real-time.")

# File uploader widget for video
uploaded_file = st.file_uploader("Choose a video", type=['mp4'])

# Function to process video with vehicle tracking
def process_video_with_tracking(video_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Temporary file for saving the output video
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')

    # Set up video writer for output video with YOLO detections and tracking
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file.name, fourcc, fps, (width, height))

    # Dictionary to store cumulative vehicle counts
    vehicle_counts = {class_names[c]: 0 for c in vehicle_classes}
    seen_ids = set()  # Track seen object IDs to avoid counting them multiple times

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Run YOLOv8 on each frame
        results = model(frame)

        # Prepare list for detected vehicles (for tracking)
        detections = []
        for result in results:
            for box in result.boxes:
                if box.cls.item() in vehicle_classes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = box.conf.item()
                    class_id = box.cls.item()
                    detections.append([x1, y1, x2, y2, confidence, class_id])

        # Update tracker with new detections
        tracked_objects = tracker.update(np.array(detections))

        # Draw bounding boxes and display vehicle types
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id, class_id = map(int, obj[:6])
            if class_id in vehicle_classes:
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_names[class_id]} {obj_id}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Count the vehicle only once by checking if the ID has been seen before
                if obj_id not in seen_ids:
                    vehicle_counts[class_names[class_id]] += 1
                    seen_ids.add(obj_id)

        # Write the frame with detections and tracking to the output video
        out.write(frame)

    # Release video resources
    cap.release()
    out.release()

    return output_file.name, vehicle_counts

# Display results when video is uploaded
if uploaded_file is not None:
    # Save video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    st.video(tfile.name)
    st.write("Detecting and tracking vehicles in real-time...")

    # Call function to process video and get the output video
    output_video_path, vehicle_counts = process_video_with_tracking(tfile.name)
    
    # Show the processed video
    st.video(output_video_path)
    
    # Display cumulative vehicle counts from the video
    st.write("**Vehicle Count in Video** (Real-Time Tracking):")
    for vehicle, count in vehicle_counts.items():
        st.write(f"{vehicle}: {count}")
