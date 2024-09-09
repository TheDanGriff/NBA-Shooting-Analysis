import sys
import cv2
import os
import matplotlib.pyplot as plt
import gdown
from openpose import pyopenpose as op  # Ensure the correct import path

# Step 1: Add OpenPose path to Python path (Adjust as per your OpenPose installation)
sys.path.append('C:\openpose\bin\python\openpose\Release')  # Adjust the path if necessary

# Step 2: Download video from Google Drive using gdown
drive_url = "https://drive.google.com/uc?id=1u58LaWMfTpVjOVUSlCxJ1Ukw9eopCkMY"
video_path = "C:/openpose/videos/first_video.mp4"
gdown.download(drive_url, video_path, quiet=False)

# Step 3: Set up OpenPose parameters and initialize OpenPose
params = dict()
params["model_folder"] = "C:/openpose/models/"  # Ensure the correct model path
params["face"] = False
params["hand"] = False
params["net_resolution"] = "656x368"

# Initialize OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Step 4: Create the output directory for processed frames
output_dir = "C:/openpose/output_frames/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Step 5: Process the first 20 seconds of the video (assuming 30 fps, that's 600 frames)
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
    
    # Process frame with OpenPose
    opWrapper.emplaceAndPop([datum])
    
    # Step 6: Check if keypoints are detected and save the frame
    if datum.poseKeypoints is not None:
        print(f"Keypoints detected for frame {frame_count + 1}")
    else:
        print(f"No keypoints detected for frame {frame_count + 1}")
    
    # Save the processed frame with keypoints overlaid
    output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(output_path, datum.cvOutputData)

    # Display the frame with keypoints using matplotlib
    frame_rgb = cv2.cvtColor(datum.cvOutputData, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(frame_rgb)
    plt.title(f"Frame {frame_count + 1}")
    plt.axis('off')
    plt.show()

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

print(f"Processed first 20 seconds of the video and saved frames in {output_dir}")

# Step 7: Test with a sample image to verify OpenPose is working
print("\nTesting with a sample image to verify OpenPose is working correctly...")

# Download a test image using requests library
image_url = "https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/examples/media/COCO_val2014_000000000192.jpg"
image_path = "C:/openpose/test_image.jpg"
gdown.download(image_url, image_path, quiet=False)

# Process the image with OpenPose
datum = op.Datum()
image_to_process = cv2.imread(image_path)
datum.cvInputData = image_to_process

opWrapper.emplaceAndPop([datum])

# Show the keypoints and the processed image
if datum.poseKeypoints is not None:
    print("Keypoints detected for test image")
    print(datum.poseKeypoints)
else:
    print("No keypoints detected for test image")

plt.imshow(cv2.cvtColor(datum.cvOutputData, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()



