import cv2
import numpy as np

def extract_frames(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_index = 0
    if not cap.isOpened():
        print("Can't open VDO:", video_path)
        return []
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # หมดวิดีโอ
        if frame_index % frame_interval == 0:
            frames.append({
                'frame_index': frame_index,
                'image': frame
            })
            print(f"frame at {frame_index}")
        frame_index += 1
    cap.release()
    return frames