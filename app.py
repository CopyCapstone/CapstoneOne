import streamlit as st
import cv2
import os
from Module_00_Frontend.extract_frames import extract_frames
from Module_00_Frontend.read_all_pixels import read_all_pixels
from Module_01_ObjectDetection.detect_and_crop import detect_and_crop 

# --- Page configuration ---
st.title("Module 1: Object Detection")
                   
# --- Helper functions ---
def save_temp_video(uploaded_video):
    """Save uploaded video once and return path."""
    project_dir = os.path.abspath(os.path.dirname(__file__))
    tmp_dir = os.path.join(project_dir, "tmp/uploaded_video")
    os.makedirs(tmp_dir, exist_ok=True)
    temp_path = os.path.join(tmp_dir, f"uploaded_video.mp4")
    with open(temp_path, "wb") as f:
        f.write(uploaded_video.read())
    return temp_path
@st.cache_resource
def get_video_capture(video_path):
    """Cache cv2.VideoCapture object for performance."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Cannot open video: {video_path}")
        return None
    return cap
@st.cache_data # Cache the return value (metadata)
def get_video_info(video_path):
    """Precompute metadata only once."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0, 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    return total_frames, fps, duration

# --- Initialize session state ---
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "stored_frame_num" not in st.session_state:
    st.session_state.stored_frame_num = 0
    
# --- Callback to update frame number ---
def update_frame_num():
    """
    Called when the slider changes.
    This updates the session state *before* the script reruns.
    'slider_frame' is the key of the st.slider widget.
    """
    st.session_state.stored_frame_num = st.session_state.slider_frame

# --- Upload section ---
uploaded_video = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
if uploaded_video:
    new_video_path = save_temp_video(uploaded_video)
    # Check if this is a NEW video
    if new_video_path != st.session_state.video_path:
        st.session_state.video_path = new_video_path
        st.session_state.stored_frame_num = 0 # Reset frame for new video
        # Clear caches for the new video
        get_video_capture.clear()
        get_video_info.clear()
    st.sidebar.success(f"Video loaded: {uploaded_video.name}")

# --- Video processing ---
if st.session_state.video_path and os.path.exists(st.session_state.video_path):
    video_path = st.session_state.video_path
    
    # Load metadata
    total_frames, fps, duration = get_video_info(video_path)

    if total_frames > 0:
        st.sidebar.video(video_path)
        st.sidebar.write(f"Duration: {duration:.2f}s ({total_frames} frames @ {fps:.1f} FPS)")

        # Frame selection slider
        st.slider(
            "Frame Number",
            0, total_frames - 1,
            value=st.session_state.stored_frame_num,
            key="slider_frame", # The key to access the slider's value in state
            on_change=update_frame_num # The callback function
        )
        # The 'stored_frame_num' is now always up-to-date
        frame_num = st.session_state.stored_frame_num

        # --- Efficient frame reading ---
        cap = get_video_capture(video_path)
        if cap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            success, frame_bgr = cap.read()
            if success:
                # 1. แสดงรูปต้นฉบับ (แปลงเป็น RGB สำหรับ Streamlit)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption=f"Frame {frame_num}")

                # 2. บันทึกเฟรมปัจจุบันเป็นไฟล์ชั่วคราว เพื่อส่ง Path ให้ detect_and_crop
                temp_frame_path = os.path.join("tmp/current_frames", f"frame_{frame_num}.jpg")
                os.makedirs("tmp/current_frames", exist_ok=True)
                cv2.imwrite(temp_frame_path, frame_bgr) # บันทึกเป็น BGR ตามมาตรฐาน OpenCV

                # 3. เรียกใช้ฟังก์ชันโดยส่ง Path เข้าไป
                # ฟังก์ชันจะส่งคืน list ของ bytes ของรูปที่ crop แล้ว
                cropped_images_bytes = detect_and_crop(temp_frame_path, pad_x_pct=0.1, pad_y_pct=0.1)

                # 4. แสดงผลรูปที่ถูก Crop
                if cropped_images_bytes:
                    cols = st.columns(len(cropped_images_bytes))
                    for idx, img_bytes in enumerate(cropped_images_bytes):
                        with cols[idx]:
                            st.image(img_bytes, caption=f"Object {idx+1}")
                else:
                    st.info("🔍 No objects detected in this frame.")
            else:
                st.error("⚠️ Could not read this frame. Try another one.")
        else:
            st.error("⚠️ Video capture object is not available.")
    else:
        st.error("⚠️ Could not process video. It may be corrupt or have 0 frames.")
else:
    st.info("👈 Please upload a video file to begin.")
