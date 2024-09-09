import sys
import cv2
import os
import matplotlib.pyplot as plt
import gdown

# Define the folder and video path
video_folder = 'C:/openpose/videos'
video_path = os.path.join(video_folder, 'nba_shooting_video.mp4')

# Ensure the directory exists
if not os.path.exists(video_folder):
    os.makedirs(video_folder)

# Step 1: Add OpenPose path to Python path (Adjust as per your OpenPose installation)
sys.path.append(r'C:\openpose\bin\python\openpose\Release')  # Adjust the path if necessary

# Step 2: Download video from Google Drive using gdown
drive_url = "https://drive.google.com/uc?id=1u58LaWMfTpVjOVUSlCxJ1Ukw9eopCkMY"
video_path = "C:/openpose/videos/first_video.mp4"
gdown.download(drive_url, video_path, quiet=False)




