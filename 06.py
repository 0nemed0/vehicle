import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model (make sure 'yolov8n.pt' is downloaded or adjust the path)
model = YOLO('yolov8n.pt')

def detect_vehicles(frame):
    # Run the YOLO model on the frame
    results = model(frame)
    return results

def draw_line(frame, line_position):
    # Draw a green line on the frame
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 255, 0), 2)

def track_vehicles(results, line_position, vehicle_ids):
    current_vehicle_ids = set()
    
    for result in results:
        for box in result.boxes.xywh:  # Bounding boxes in [x_center, y_center, width, height] format
            x_center, y_center = int(box[0]), int(box[1])

            # Generate a simple ID for each detected vehicle based on its position
            object_id = hash(f'{x_center}{y_center}')
            
            if object_id not in vehicle_ids and y_center > line_position:
                # If the vehicle crosses the line for the first time
                vehicle_ids.add(object_id)
                current_vehicle_ids.add(object_id)

    return current_vehicle_ids

def main():
    st.title("Real-time Vehicle Counting System")

    # Video source (can be a file or webcam)
    video = cv2.VideoCapture(0)  # Replace '0' with the path to a video file if needed

    line_position = 300  # Y-coordinate of the counting line
    vehicle_ids = set()  # A set to store vehicle IDs that have crossed the line
    vehicle_count = 0

    stframe = st.empty()  # Placeholder for video frame

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        # Detect vehicles in the frame
        results = detect_vehicles(frame)

        # Draw the counting line
        draw_line(frame, line_position)

        # Track vehicles crossing the line
        new_vehicles = track_vehicles(results, line_position, vehicle_ids)
        vehicle_count += len(new_vehicles)

        # Display the current frame and count
        stframe.image(frame, channels="BGR")
        st.write(f"Vehicle Count: {vehicle_count}")

    video.release()

if __name__ == "__main__":
    main()