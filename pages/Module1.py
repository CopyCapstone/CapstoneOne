import streamlit as st
import cv2
import os
from pathlib import Path
from Module_01_ObjectDetection.detect_rotate_crop import detect_rotate_crop

# --- Configuration & Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
TMP_DIR = PROJECT_ROOT / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# --- Save Temp Video functions ---
def save_temp_video(uploaded_file):
    video_dir = TMP_DIR / "uploaded_video"
    video_dir.mkdir(parents=True, exist_ok=True)
    # Use the actual filename to prevent overwriting issues during a single session
    file_path = video_dir / uploaded_file.name 
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)

# Remove @st.cache_resource here as it often causes "File closed" errors with OpenCV
def get_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    return cap

@st.cache_data
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0, 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Keep the -2 offset if that's specific to your video encoding needs
    total_frames = max(0, total_frames - 2) 
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    return total_frames, fps, duration

# --- Initialize session state ---
def init_session_state():
    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    if "stored_frame_num" not in st.session_state:
        st.session_state.stored_frame_num = 0
    if "stored_pad_x" not in st.session_state:
        st.session_state.stored_pad_x = -0.1
    if "stored_pad_y" not in st.session_state:
        st.session_state.stored_pad_y = -0.05
    if "stored_shrink" not in st.session_state:
        st.session_state.stored_shrink = 0.2

init_session_state()

st.title("🔍 Module 1: Object Detection")
st.divider()

# --- Sidebar UI ---
with st.sidebar:
    st.header("📽️ Video Input")
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    
    if uploaded_video:
        new_path = save_temp_video(uploaded_video)
        # Only reset if the path (filename) is actually different
        if new_path != st.session_state.video_path:
            st.session_state.video_path = new_path
            st.session_state.stored_frame_num = 0
            # Clear info cache for the new file
            get_video_info.clear()
            st.rerun() # Refresh to ensure clean state for new video

# --- Main Logic ---
if st.session_state.video_path:
    v_path = st.session_state.video_path
    total_frames, fps, duration = get_video_info(v_path)

    with st.sidebar:
        if total_frames > 0:
            st.divider()
            st.write(f"🎞️ {total_frames} Frames | {duration:.2f}s")
            
            # Use value= to sync, but key= for internal state
            frame_sel = st.slider("Select Frame", 0, total_frames, 
                                 value=st.session_state.stored_frame_num)
            st.session_state.stored_frame_num = frame_sel

            with st.expander("⚙️ Detection Settings"):
                st.session_state.stored_pad_x = st.slider("Padding X", -0.45, 0.45, st.session_state.stored_pad_x)
                st.session_state.stored_pad_y = st.slider("Padding Y", -0.45, 0.45, st.session_state.stored_pad_y)
                st.session_state.stored_shrink = st.slider("Shrink", 0.0, 0.45, st.session_state.stored_shrink)

    # Process and Display
    cap = get_video_capture(v_path)
    if cap:
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.stored_frame_num)
        success, frame_bgr = cap.read()
        cap.release() # Release immediately after reading

        if success:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.subheader("Detected")
                cropped = detect_rotate_crop(frame_bgr, 
                                            pad_x_pct=st.session_state.stored_pad_x, 
                                            pad_y_pct=st.session_state.stored_pad_y, 
                                            shrink=st.session_state.stored_shrink)
                if cropped is not None:
                    st.image(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    st.warning("No object detected.")
else:
    st.info("👈 Please upload a video to start.")