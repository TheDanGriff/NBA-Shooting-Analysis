import video_processing
import pose_estimation
import ball_tracking
import sync_data
import model_training
import visualization

def run_pipeline():
    # Step 1: Authenticate and download video from Google Drive
    print("Downloading video from Google Drive...")
    drive = video_processing.authenticate_gdrive()
    video_file_name = video_processing.list_and_download_latest_video(drive, './')
    
    if not video_file_name:
        print("No video downloaded. Exiting.")
        return
    
    # Step 2: Run pose estimation on the downloaded video
    print("Running pose estimation...")
    pose_estimation.run_openpose(video_file_name, './pose_data')

    # Step 3: Run YOLO and DeepSORT for ball tracking
    print("Running ball tracking...")
    ball_tracking.run_yolo(video_file_name)
    ball_tracking.run_deepsort(video_file_name)

    # Step 4: Synchronize pose and ball tracking data
    print("Synchronizing data...")
    sync_data.sync_pose_and_ball_data('./pose_data', './yolov5/runs/detect/exp/labels/input_video.csv')

    # Step 5: Train machine learning model
    print("Training model...")
    model_training.train_shot_prediction_model('merged_shot_data.csv')

    # Step 6: Visualize the results
    print("Visualizing results...")
    visualization.visualize_trajectory_and_pose(video_file_name, 'merged_shot_data.csv')

if __name__ == "__main__":
    run_pipeline()
