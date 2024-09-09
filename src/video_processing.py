import cv2
import sys
import os
from openpose import pyopenpose as op

# Output directory for the processed frames
output_dir = "/content/openpose_output/"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Configure OpenPose
params = dict()
params["model_folder"] = "/content/openpose/models/"
params["face"] = False
params["hand"] = False
params["net_resolution"] = "320x176"

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Read the video from Google Drive
cap = cv2.VideoCapture(video_path)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    
    # Save the processed frame for debugging purposes
    output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(output_path, datum.cvOutputData)
    frame_count += 1

cap.release()
cv2.destroyAllWindows()

print(f"Processed video saved in {output_dir}")
