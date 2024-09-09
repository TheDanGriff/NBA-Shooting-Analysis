from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

def authenticate_gdrive():
    # Authenticate Google Drive API
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # This creates a local server to authenticate
    drive = GoogleDrive(gauth)
    return drive

def list_and_download_latest_video(drive, output_path):
    # List all video files in your Google Drive
    file_list = drive.ListFile({'q': "mimeType contains 'video/'"}).GetList()
    
    if not file_list:
        print("No video files found in Google Drive.")
        return None
    
    # Select the most recent video
    video_file = file_list[0]  # Change this logic if needed to get a specific file
    for file in file_list:
        print(f"Title: {file['title']}, ID: {file['id']}")
    
    # Download the video
    print(f"Downloading: {video_file['title']}")
    video_file.GetContentFile(os.path.join(output_path, video_file['title']))
    return video_file['title']

if __name__ == "__main__":
    output_path = "./"  # Directory where you want to save the video
    drive = authenticate_gdrive()
    video_file_name = list_and_download_latest_video(drive, output_path)
    
    if video_file_name:
        print(f"Video downloaded and saved as: {video_file_name}")
    else:
        print("Failed to download video.")

