import sys
sys.path.append('/content/openpose/build/python')
import cv2
import sys
import os
from openpose import pyopenpose as op
import matplotlib.pyplot as plt  # Import matplotlib for plotting frames


# Step 3: Define the video path (already downloaded to /content)
video_path = "/content/first_video.mp4"

# Step 4: Create the output directory for processed frames
output_dir = "/content/openpose_output/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Step 5: Configure OpenPose
params = dict()
params["model_folder"] = "/content/openpose/models/"
params["face"] = False
params["hand"] = False
params["net_resolution"] = "656x368"

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Step 6: Process the first 20 seconds of the video (assuming 30 fps, that's 600 frames)
cap = cv2.VideoCapture(video_path)
frame_rate = 30  # Adjust this according to your video's actual frame rate if necessary
max_frames = 20 * frame_rate  # Number of frames for the first 20 seconds

frame_count = 0
while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    datum = op.Datum()
    datum.cvInputData = frame
    
    # Wrap datum in a list of shared pointers
    datums_pointer = op.VectorDatum()
    datums_pointer.append(datum)

    # Now emplace and pop the properly formatted list
    opWrapper.emplaceAndPop(datums_pointer)
    print(datum.poseKeypoints)

    
    # Save the processed frame
    output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(output_path, datum.cvOutputData)
    
    # Use matplotlib to display the frame
    frame_rgb = cv2.cvtColor(datum.cvOutputData, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(frame_rgb)
    plt.title(f"Frame {frame_count + 1}")
    plt.axis('off')
    
    # Show the plot to display it inline
    plt.show()

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

print(f"Processed first 20 seconds of the video and saved in {output_dir}")





