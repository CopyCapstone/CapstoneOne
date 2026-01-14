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


def read_all_pixels(frame):
    if not isinstance(frame, np.ndarray):
        raise ValueError("frame must be a numpy array")

    # แปลง BGR → RGB
    frame_rgb = frame[:, :, ::-1]

    h, w, c = frame_rgb.shape
    pixels = frame_rgb.copy()

    # หาค่าเฉลี่ยสี
    avg_color = np.mean(pixels.reshape(-1, 3), axis=0)
    avg_color = np.round(avg_color, 3)

    # ✅ แปลงเป็น float ปกติ เพื่อไม่ให้แสดง np.float64(...)
    avg_color = tuple(float(v) for v in avg_color)

    print(f"Frame size: {w}x{h} px, avg RGB: {avg_color}")
    return pixels, avg_color