import sys
import cv2

# Add the path to the directory containing pyopenpose.cp37-win_amd64.pyd
sys.path.append("C:/openpose/bin/python/openpose/Release")  # Adjust this path as needed

# Now import OpenPose
import pyopenpose as op

def run_openpose(video_path, output_dir):
    # Configure OpenPose parameters
    params = dict()
    params["model_folder"] = "C:/openpose/models/"

    # Initialize OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Load the video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])

        # Save keypoints and the processed frame
        cv2.imwrite(f"{output_dir}/frame_{frame_count}_pose.jpg", datum.cvOutputData)
        print(datum.poseKeypoints)  # Print the keypoints for the frame
        frame_count += 1

    cap.release()

if __name__ == "__main__":
    video_path = "AdobeStock_499964336_Video_4K_Preview.mov"  # Your video file
    output_dir = "C:/openpose/output/pose_data"
    run_openpose(video_path, output_dir)


