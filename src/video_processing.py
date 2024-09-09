import sys
sys.path.append('/content/openpose/build/python')
import cv2
import os
import matplotlib.pyplot as plt
from openpose import pyopenpose as op

# Step 1: Verify Model Path and Configuration
params = dict()
params["model_folder"] = "/content/openpose/models/"  # Ensure the correct path
params["face"] = False
params["hand"] = False
params["net_resolution"] = "656x368"  # Increase resolution for better accuracy

# Initialize OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Step 2: Define the video path (already downloaded to /content)
video_path = "/content/first_video.mp4"  # Adjust to your video path

# Step 3: Create the output directory for processed frames
output_dir = "/content/openpose_output/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Step 4: Process the first 20 seconds of the video (assuming 30 fps, that's 600 frames)
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
    
    # Step 5: Debugging - Check if keypoints are detected
    if datum.poseKeypoints is not None:
        print(f"Keypoints detected for frame {frame_count + 1}")
        print(datum.poseKeypoints)
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

print(f"Processed first 20 seconds of the video and saved in {output_dir}")

# Step 6: Test with a sample image to verify OpenPose is working
print("\nTesting with a sample image to verify OpenPose is working correctly...")

# Download a test image
!wget -O test_image.jpg https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/examples/media/COCO_val2014_000000000192.jpg

# Process the image with OpenPose
image_path = "/content/test_image.jpg"
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


