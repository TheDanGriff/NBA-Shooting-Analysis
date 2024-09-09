import sys
import os
import cv2

# Add OpenPose library to the Python path
sys.path.append(r'C:\openpose\openpose\bin\python\openpose\Release')
os.environ['PATH'] += ';C:/openpose/openpose/bin/x64/Release;C:/openpose/openpose/bin;'

try:
    from openpose import pyopenpose as op
except ImportError as e:
    print("Error: OpenPose library could not be found. Ensure the library is set up correctly.")
    raise e

# Step 1: Configure OpenPose parameters
params = dict()
params["model_folder"] = "C:/openpose/openpose/models/"
params["face"] = False
params["hand"] = False

# Step 2: Initialize OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Step 3: Process the video
video_path = "C:/openpose/openpose/videos/first_video.mp4"
output_dir = "C:/openpose/openpose/openpose_output/"

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

print(f"Processed {frame_count} frames and saved in {output_dir}")


