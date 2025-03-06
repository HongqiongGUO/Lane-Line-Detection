import cv2
import time
import numpy as np
from LLD5_Lane_Detection_Pipeline import lane_finding_pipeline

# Set the path of the input and ouput video
input_video_path = "CarND-LaneLines-P1/test_videos/solidWhiteRight.mp4"
output_video_path = "solidWhiteRight_fixed.mp4"

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Get the parameters of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create video writing object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 encoding format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"Processing Video: {input_video_path} -> {output_video_path}")
print(f"FPS: {fps}, Resolution: {width}x{height}, Total Frames: {frame_count}")

# Record start time
start_time = time.time()

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Read end

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format
    processed_frame = lane_finding_pipeline(frame)  # Processing frame
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR
    out.write(processed_frame)  # Write a new video
    
    frame_idx += 1
    if frame_idx % 50 == 0:
        print(f"Processed {frame_idx}/{frame_count} frames...")

# Release resources
cap.release()
out.release()

# Record end time
end_time = time.time()
print(f"Video processing completed! Total time of {end_time - start_time:.2f} s")
