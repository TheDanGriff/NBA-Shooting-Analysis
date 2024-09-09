import sys
import cv2
import os
from openpose import pyopenpose as op
# Include the OpenPose library path (adjust the path to your specific setup)
import sys
sys.path.append('C:/openpose/build/python/openpose/Release')  # Update this if needed



# Step 1: Configure OpenPose parameters
params = dict()
params["model_folder"] = "C:/openpose/models/"
params["face"] = False
params["hand"] = False

# Step 2: Initialize OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Step 3: Process the video
video_path = "nba_shooting_video.mp4"
output_dir = "openpose_output/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(output_path, datum.cvOutputData)

    frame_count += 1

cap.release()
cv2.destroyAllWindows()


