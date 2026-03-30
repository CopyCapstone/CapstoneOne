import streamlit as st
import cv2
import os
from pathlib import Path
from Module_01_ObjectDetection.detect_rotate_crop import detect_rotate_crop

# --- Configuration & Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
TMP_DIR = PROJECT_ROOT / "tmp"

# --- Save Temp Video functions ---
def save_temp_video(uploaded_file):
    video_dir = TMP_DIR / "uploaded_video"
    video_dir.mkdir(parents=True, exist_ok=True)
    file_path = video_dir / "uploaded_video.mp4"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)

@st.cache_resource
def get_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Cannot open video: {video_path}")
        return None
    return cap

@st.cache_data
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0, 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = total_frames -2
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    return total_frames, fps, duration
    
# --- Initialize session state ---
def init_session_state():
    """รวมการประกาศ Session State """
    defaults = {
        "video_path": None,
        "stored_frame_num": 0,
        "stored_pad_x": -0.1,
        "stored_pad_y": -0.05,
        "stored_shrink": 0.2
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val  
init_session_state()

# --- Main Display Area ---
st.title("🔍 Module 1: Object Detection")
st.divider()

# --- Sidebar UI Upload section ---
total_frames, fps, duration = 0, 0, 0 # Default values in case no video is loaded
with st.sidebar:
    st.header("📽️ Video Input")
    uploaded_video = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi"], help="Supported Formats: MP4, MOV, AVI (Max 200 MB)")
    if uploaded_video:
        # Clear caches for the new video
        get_video_capture.clear()
        get_video_info.clear()
        new_video_path = save_temp_video(uploaded_video)
        # Check if this is a NEW video
        if new_video_path != st.session_state.video_path:
            st.session_state.video_path = new_video_path
            st.session_state.stored_frame_num = 0 # Reset frame for new video

    if st.session_state.video_path:
        if not uploaded_video:
            st.success(f"Using loaded video: {os.path.basename(st.session_state.video_path)}")
        total_frames, fps, duration = get_video_info(st.session_state.video_path)
        if total_frames > 0:
            st.divider()
            st.markdown(f"Info: {duration:.2f}s | FPS: {fps} | {total_frames} frames")
            col_f1, col_f2 = st.columns([0.7, 0.3])
            
            with col_f1:
                slider_val = st.slider(
                    "Select Frame", 0, total_frames,
                    value=st.session_state.stored_frame_num,
                )
                if slider_val != st.session_state.stored_frame_num:
                    st.session_state.stored_frame_num = slider_val
                    st.rerun()
                    
            with col_f2:
                slider_val = st.number_input(
                    "Frame", 0, total_frames,
                    value=st.session_state.stored_frame_num,
                )
                if slider_val != st.session_state.stored_frame_num:
                    st.session_state.stored_frame_num = slider_val
                    st.rerun()


            with st.expander("⚙️ Detection Settings", expanded=False):
                st.subheader("Default Pad_X = -0.1 Pad_Y = -0.05 Shrink = 0.2", width='stretch')
                pad_x_slider = st.slider("Padding X", -0.45, 0.45, step=0.01, value= st.session_state.stored_pad_x)
                pad_y_slider = st.slider("Padding Y", -0.45, 0.45, step=0.01, value= st.session_state.stored_pad_y)
                shrink_slider = st.slider("Final Shrink", 0.0, 0.45, step=0.01, value= st.session_state.stored_shrink)
                if (pad_x_slider != st.session_state.stored_pad_x or 
                    pad_y_slider != st.session_state.stored_pad_y or 
                    shrink_slider != st.session_state.stored_shrink):
                    st.session_state.stored_pad_x = pad_x_slider
                    st.session_state.stored_pad_y = pad_y_slider
                    st.session_state.stored_shrink = shrink_slider
                    st.rerun()

if st.session_state.video_path and os.path.exists(st.session_state.video_path):
    cap = get_video_capture(st.session_state.video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.stored_frame_num)
    success, frame_bgr = cap.read()  
    if success:
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            st.markdown(f"### 🖼️ Original Frame {st.session_state.stored_frame_num}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, width='stretch')
            # --- Debug Section (Optional) ---
            debug_dir = TMP_DIR / "video_frames"
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_path = debug_dir / f"frame_{st.session_state.stored_frame_num}.jpg"
            cv2.imwrite(str(debug_path), frame_bgr)
        with col2:
            st.markdown(f"### 🎯 Detected Frame {st.session_state.stored_frame_num}")
            pad_x = st.session_state.get("pad_x_slider", -0.1)
            pad_y = st.session_state.get("pad_y_slider", -0.05)
            shrink_val = st.session_state.get("shrink_slider", 0.2)
            cropped_result = detect_rotate_crop(frame_bgr, pad_x_pct=pad_x, pad_y_pct=pad_y, shrink=shrink_val)
            if cropped_result is not None:
                result_rgb = cv2.cvtColor(cropped_result, cv2.COLOR_BGR2RGB)
                st.image(result_rgb, width='stretch')
                # --- Debug Section (Optional) ---
                debug_dir = TMP_DIR / "cropped_frames"
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_path = debug_dir / f"cropped_frame_{st.session_state.stored_frame_num}.jpg"
                cv2.imwrite(str(debug_path), cropped_result)
            else:
                st.warning("No object detected in this frame.")
        with st.expander("ℹ️ Settings Details"):
            st.write(f"Processing Frame: {st.session_state.stored_frame_num}")
            st.write(f"Object Detection Pad_x: {st.session_state.stored_pad_x}")
            st.write(f"Object Detection Pad_y: {st.session_state.stored_pad_y}")
            st.write(f"Object Detection Shrink: {st.session_state.stored_shrink}")
    else:
        st.error("Failed to read the selected frame.")
else:
    st.info("👈 Please upload a video from the sidebar to begin.")