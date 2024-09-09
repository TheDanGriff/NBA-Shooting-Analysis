import pandas as pd

def sync_pose_and_ball_data(pose_json_dir, ball_tracking_csv):
    pose_data = pd.read_json(f'{pose_json_dir}/pose_1.json')  # First frame as an example
    ball_data = pd.read_csv(ball_tracking_csv)
    
    # Merge based on frame number
    merged_data = pd.merge(pose_data, ball_data, on='frame')
    
    # Save synchronized data
    merged_data.to_csv('merged_shot_data.csv', index=False)
    return merged_data

if __name__ == "__main__":
    pose_json_dir = "C:/path/to/pose_data"
    ball_tracking_csv = "C:/path/to/yolov5/runs/detect/exp/labels/input_video.csv"
    merged_data = sync_pose_and_ball_data(pose_json_dir, ball_tracking_csv)
    print("Synchronized data saved as 'merged_shot_data.csv'")
