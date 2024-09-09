import sys
import cv2
import os

# Include the OpenPose library path
sys.path.append('C:/openpose/openpose/bin/python/openpose/Release')  # Adjusted to your pyopenpose path
os.environ['PATH'] += ';C:/openpose/openpose/x64/Release;'  # Add OpenPose binaries to the system PATH

try:
    from openpose import pyopenpose as op
except ImportError as e:
    print("Error: OpenPose library could not be found. Ensure the library is set up correctly.")
    raise e

# Step 1: Configure OpenPose parameters
params = dict()
params["model_folder"] = "C:/openpose/openpose/models/"  # Path to your models folder
params["face"] = False
params["hand"] = False
params["net_resolution"] = "656x368"  # Set net resolution to enhance detection accuracy

# Step 2: Initialize OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Step 3: Process the video (Ensure your video is correctly placed in the path specified)
video_path = "C:/openpose/openpose/videos/first_video.mp4"  # Path to the video
output_dir = "C:/openpose/openpose/openpose_output/"  # Output directory for processed frames

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Step 4: Capture the video and run OpenPose
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

print(f"Processed {frame_count} frames and saved in {output_dir}")


