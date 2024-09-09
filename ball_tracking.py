import subprocess

def run_yolo(video_path):
    subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5'])
    subprocess.run(['pip', 'install', '-r', 'yolov5/requirements.txt'])
    subprocess.run(['pip', 'install', 'deep_sort_realtime'])

    # Run YOLOv5 on the video for ball detection
    detect_command = [
        'python', 'yolov5/detect.py',
        '--source', video_path,
        '--weights', 'yolov5s.pt',
        '--save-txt',
        '--save-conf',
        '--exist-ok'
    ]
    subprocess.run(detect_command)

def run_deepsort(video_path):
    subprocess.run(['git', 'clone', 'https://github.com/nwojke/deep_sort.git'])
    track_command = [
        'python', 'deep_sort/track.py',
        '--source', video_path
    ]
    subprocess.run(track_command)

if __name__ == "__main__":
    video_path = "AdobeStock_499964336_Video_4K_Preview.mov"
    run_yolo(video_path)
    run_deepsort(video_path)
