from CV.extract_frames import read_all_pixels, extract_frames

if __name__ == "__main__":
    video_path = "video/sample_video.mp4"

    frames = extract_frames(video_path, frame_interval=20)

    for info in frames:
        frame = info['image']
        pixels, avg_color = read_all_pixels(frame)
        print(f"Frame {info['frame_index']} → RGB = {avg_color}\n")
