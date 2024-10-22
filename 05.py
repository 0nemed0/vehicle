import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image
import tempfile
import math

# Load YOLOv8 model (pre-trained)
model = YOLO('yolov8n.pt')

# Vehicle class IDs from the COCO dataset
vehicle_classes = [2, 3, 5, 7]  # 2: car, 3: motorcycle, 5: bus, 7: truck
class_names = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}

# Initialize the DeepSort tracker
tracker = DeepSort(max_age=5, n_init=3, max_iou_distance=0.7)

# Streamlit app title and instructions
st.title('Real-Time Vehicle Detection and Counting Across Parallel Lines')

# Option for user to select between video file and webcam
option = st.selectbox("Select input source:", ("Upload Video", "Live Webcam"))

# Adjusting two parallel lines (their start points and angle)
st.write("Adjust the two parallel lines for vehicle counting.")

# Line position sliders (line1_start and line2_start are the start points of the two lines)
line1_x = st.slider("Line 1 - X Coordinate", min_value=0, max_value=1000, value=100)
line1_y = st.slider("Line 1 - Y Coordinate", min_value=0, max_value=1000, value=200)
line_angle = st.slider("Line Angle (degrees)", min_value=0, max_value=180, value=0)

# Set line length and calculate endpoints for two parallel lines
line_length = 500
line1_end_x = line1_x + int(line_length * math.cos(math.radians(line_angle)))
line1_end_y = line1_y + int(line_length * math.sin(math.radians(line_angle)))
line2_x = line1_x + 100  # Parallel line (100px apart)
line2_end_x = line2_x + int(line_length * math.cos(math.radians(line_angle)))
line2_end_y = line1_y + int(line_length * math.sin(math.radians(line_angle)))

# Function to check if a vehicle crosses a line (between start and end points)
def is_vehicle_crossing_line(x1, y1, x2, y2, p_x, p_y):
    line_len = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    distance_to_line = abs((y2 - y1)*p_x - (x2 - x1)*p_y + x2*y1 - y2*x1) / line_len
    return distance_to_line < 10  # Adjust this threshold for line crossing precision

# Function to process video with vehicle tracking and counting based on crossing lines
def process_video_with_lines(video_path):
    cap = cv2.VideoCapture(video_path)
    return process_video_stream_with_lines(cap)

# Function to process webcam stream with tracking and counting based on crossing lines
def process_webcam_with_lines():
    cap = cv2.VideoCapture(0)  # 0 is the default webcam
    return process_video_stream_with_lines(cap)

# Common function to process a video stream (video file or webcam) and count vehicles across two lines
def process_video_stream_with_lines(cap):
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

    vehicles_crossed_first_line = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw parallel lines on the frame
        cv2.line(frame, (line1_x, line1_y), (line1_end_x, line1_end_y), (0, 0, 255), 2)
        cv2.line(frame, (line2_x, line1_y), (line2_end_x, line2_end_y), (0, 255, 0), 2)

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
        tracked_objects = tracker.update_tracks(detections, frame=frame)

        # Draw bounding boxes and display vehicle types
        for track in tracked_objects:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)
            class_id = track.class_id  # Assign appropriate class_id here

            if class_id in vehicle_classes:
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_names[class_id]} {track_id}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Check if the vehicle crosses the first and then second line
                centroid_x = (x1 + x2) // 2
                centroid_y = (y1 + y2) // 2

                if is_vehicle_crossing_line(line1_x, line1_y, line1_end_x, line1_end_y, centroid_x, centroid_y):
                    vehicles_crossed_first_line.add(track_id)
                if track_id in vehicles_crossed_first_line and is_vehicle_crossing_line(line2_x, line1_y, line2_end_x, line2_end_y, centroid_x, centroid_y):
                    if track_id not in seen_ids:
                        vehicle_counts[class_names[class_id]] += 1
                        seen_ids.add(track_id)

        # Write the frame with detections and tracking to the output video
        out.write(frame)

    # Release video resources
    cap.release()
    out.release()

    return output_file.name, vehicle_counts

# If user selects to upload a video
if option == "Upload Video":
    uploaded_file = st.file_uploader("Choose a video", type=['mp4'])
    if uploaded_file is not None:
        # Save video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        st.video(tfile.name)
        st.write("Detecting and tracking vehicles in the uploaded video...")
        
        # Process video file with tracking and line-based counting
        output_video_path, vehicle_counts = process_video_with_lines(tfile.name)
        
        # Show the processed video
        st.video(output_video_path)
        
        # Display cumulative vehicle counts
        st.write("**Vehicle Count in Video** (Based on Line Crossings):")
        for vehicle, count in vehicle_counts.items():
            st.write(f"{vehicle}: {count}")

# If user selects to use the webcam
elif option == "Live Webcam":
    st.write("Starting live webcam for real-time vehicle detection and tracking...")
    
    # Process webcam stream with tracking and line-based counting
    output_video_path, vehicle_counts = process_webcam_with_lines()
    
    # Display cumulative vehicle counts
    st.write("**Vehicle Count in Live Webcam** (Based on Line Crossings):")
    for vehicle, count in vehicle_counts.items():
        st.write(f"{vehicle}: {count}")
