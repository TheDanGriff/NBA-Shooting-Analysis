import sys
import cv2
import os

# Include the OpenPose library path (adjust the path to your specific setup)
sys.path.append('C:\openpose\openpose')  # Make sure this path is correct for your setup

# Import OpenPose
from openpose import pyopenpose as op

# Step 1: Configure OpenPose parameters
params = dict()
params["model_folder"] = "C:/openpose/models/"  # Path to OpenPose models
params["face"] = False  # Set to True if face keypoint detection is needed
params["hand"] = False  # Set to True if hand keypoint detection is needed

# Step 2: Initialize OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Step 3: Process the video
video_path = "C:/openpose/videos/nba_shooting_video.mp4"  # Adjust to your actual video path
output_dir = "C:/openpose/openpose_output/"  # Folder to save the output frames

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(video_path)
frame_count = 0

# Step 4: Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Create a new OpenPose Datum object
    datum = op.Datum()
    datum.cvInputData = frame

    # Run OpenPose on the frame
    opWrapper.emplaceAndPop([datum])

    # Save the frame with keypoints drawn
    output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(output_path, datum.cvOutputData)

    print(f"Processed frame {frame_count}")
    frame_count += 1

# Release video capture and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()

print(f"Processed {frame_count} frames and saved in {output_dir}")

