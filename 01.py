import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import tempfile

# Load YOLOv8 model (pre-trained)
model = YOLO('yolov8n.pt')

# Vehicle class IDs from the COCO dataset
vehicle_classes = [2, 3, 5, 7]  # 2: car, 3: motorcycle, 5: bus, 7: truck
class_names = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}

# Streamlit app title and instructions
st.title('Vehicle Detection and Counting App')
st.write("Upload an image or a video file to detect and count vehicles by type.")

# File uploader widget for image or video
uploaded_file = st.file_uploader("Choose an image or video", type=['jpg', 'jpeg', 'png', 'mp4'])

# Function to process image and count vehicles
def process_image(image):
    img_np = np.array(image)
    # Detect vehicles using YOLOv8
    results = model(img_np)

    # Extract detections and count vehicle types
    vehicle_counts = {class_names[c]: 0 for c in vehicle_classes}
    for result in results:
        for box in result.boxes:
            if box.cls.item() in vehicle_classes:
                vehicle_counts[class_names[box.cls.item()]] += 1

    # Plot the results (bounding boxes, labels, etc.)
    result_img = results[0].plot()
    return result_img, vehicle_counts

# Function to process video and count vehicles
def process_video(video_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Temporary file for saving the output video
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')

    # Set up video writer for output video with YOLO detections
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file.name, fourcc, fps, (width, height))

    # Dictionary to store cumulative vehicle counts
    cumulative_counts = {class_names[c]: 0 for c in vehicle_classes}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Run YOLOv8 on each frame
        results = model(frame)

        # Count vehicles in each frame
        for result in results:
            for box in result.boxes:
                if box.cls.item() in vehicle_classes:
                    cumulative_counts[class_names[box.cls.item()]] += 1

        # Draw bounding boxes on frame
        result_frame = results[0].plot()
        # Write the frame with detections to the output video
        out.write(result_frame)

    # Release video resources
    cap.release()
    out.release()

    return output_file.name, cumulative_counts

# Display results based on the file type
if uploaded_file is not None:
    file_type = uploaded_file.type
    if file_type in ["image/jpeg", "image/png", "image/jpg"]:
        # Process the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Detecting and counting vehicles in the image...")
        
        # Call function to process image and show result
        result_img, vehicle_counts = process_image(image)
        st.image(result_img, caption="Detected Vehicles", use_column_width=True)
        
        # Display vehicle counts
        st.write("**Vehicle Count in Image**:")
        for vehicle, count in vehicle_counts.items():
            st.write(f"{vehicle}: {count}")

    elif file_type == "video/mp4":
        # Save video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        st.video(tfile.name)
        st.write("Detecting and counting vehicles in the video...")

        # Call function to process video and get the output video
        output_video_path, vehicle_counts = process_video(tfile.name)
        
        # Show the processed video
        st.video(output_video_path)
        
        # Display cumulative vehicle counts from the video
        st.write("**Cumulative Vehicle Count in Video**:")
        for vehicle, count in vehicle_counts.items():
            st.write(f"{vehicle}: {count}")
