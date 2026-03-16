import streamlit as st
import cv2
import os
import numpy as np
from pathlib import Path
from skimage.color import rgb2xyz, xyz2rgb
from colour.colorimetry import CCS_ILLUMINANTS

from Module_01_ObjectDetection.detect_rotate_crop import detect_rotate_crop
from Module_02_ChromaticAdaptationTransform.cat import chromatic_adaptation_transform
from Module_02_ChromaticAdaptationTransform.custom_CAT import custom_CAT

# --- Configuration & Constants ---
st.set_page_config(page_title="Capstone Project", layout="wide")
PROJECT_ROOT = Path(__file__).parent.parent
TMP_DIR = PROJECT_ROOT / "tmp"
ILLUMINANTS_xy = CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
# --- Helper Functions (Logic) ---
@st.cache_data(show_spinner=False)
def process_cat_logic(image_path, method, lower, upper, src_light, tgt_light):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None
    if method == "custom":
        # 1. เตรียมค่า xy ของแสง
        xy_src = ILLUMINANTS_xy[src_light]
        xy_tgt = ILLUMINANTS_xy[tgt_light]
        # 2. แปลง BGR -> RGB -> XYZ (ใช้ skimage.color)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_xyz = rgb2xyz(img_rgb)
        # 3. เรียกใช้ฟังก์ชัน custom_CAT
        processed_xyz = custom_CAT(img_xyz, xy_src, xy_tgt, method='Von Kries')
        # 4. แปลง XYZ กลับเป็น RGB -> BGR
        processed_rgb_float = xyz2rgb(processed_xyz)
        processed_rgb_float = np.clip(processed_rgb_float, 0, 1)
        processed_rgb_uint8 = (processed_rgb_float * 255).astype(np.uint8)
        return cv2.cvtColor(processed_rgb_uint8, cv2.COLOR_RGB2BGR)
    else:
        return chromatic_adaptation_transform(str(image_path), method, [lower, upper])
def init_session_state():
    """รวมการประกาศ Session State """
    defaults = {
        "stored_cat_method": "grey_world",
        "stored_lower": 95,
        "stored_upper": 99,
        "stored_light_source": 'D65',
        "stored_light_target": 'D65',
        "stored_frame_num": 0
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val  
               
# --- UI Components ---  
def sidebar_controls():
    st.sidebar.header("🎨 CAT Settings")
    method_options = ["grey_world", "white_patch", "custom"]
    
    st.sidebar.segmented_control(
        "Correction Method", 
        options=method_options, 
        default=st.session_state.stored_cat_method,
        key="temp_method",
        on_change=lambda: st.session_state.update({"stored_cat_method": st.session_state.temp_method})
    )
    
    if st.session_state.stored_cat_method == "white_patch":
        st.session_state.stored_lower = st.sidebar.slider(
            "Lower Percentile", 0, 100, 
            value=st.session_state.stored_lower
        )
        st.session_state.stored_upper = st.sidebar.slider(
            "Upper Percentile", 0, 100, 
            value=st.session_state.stored_upper
        )
        
    if st.session_state.stored_cat_method == "custom":
        illuminant_list = list(ILLUMINANTS_xy.keys())
        src_idx = illuminant_list.index(st.session_state.stored_light_source)
        tgt_idx = illuminant_list.index(st.session_state.stored_light_target)
        st.session_state.stored_light_source = st.sidebar.selectbox(
            "Source (Original)", illuminant_list, 
            index=src_idx
        )
        st.session_state.stored_light_target = st.sidebar.selectbox(
            "Target (Desired)", illuminant_list, 
            index=tgt_idx
        )
        
# --- Main Application Execution ---
def main():
    st.title("🌓 Module 2: Chromatic Adaptation")
    st.divider()
    init_session_state()
    sidebar_controls()
    frame_num = st.session_state.stored_frame_num
    # กำหนด Paths
    crop_path = TMP_DIR / "cropped_frames" / f"cropped_frame_{frame_num}.jpg"
    full_path = TMP_DIR / "video_frames" / f"frame_{frame_num}.jpg"
    save_dir = TMP_DIR / "CAT_frames"
    save_dir.mkdir(parents=True, exist_ok=True)
    # ดึงค่า Setting อาจได้ใช้ 
    m1_pad_x = st.session_state.get("stored_pad_x", -0.1)
    m1_pad_y = st.session_state.get("stored_pad_y", -0.05)
    m1_shrink = st.session_state.get("stored_shrink", 0.2)
    if not full_path.exists():
        st.warning(f"⚠️ No source frame found for Frame {frame_num}. Please process Module 1 first.")
        return
    with st.spinner("✨ Applying Color Adaptation..."):
        # รัน CAT โดยเลือกวิธีการตามที่ผู้ใช้กำหนด
        if st.session_state.stored_cat_method == "custom":
            processed_bgr = process_cat_logic(
                crop_path, 
                st.session_state.stored_cat_method,
                st.session_state.stored_lower,
                st.session_state.stored_upper,
                st.session_state.stored_light_source,
                st.session_state.stored_light_target
            )        
        else:
            processed_full_bgr = process_cat_logic(
                full_path, 
                st.session_state.stored_cat_method,
                st.session_state.stored_lower,
                st.session_state.stored_upper,
                None,
                None
            )            
            if processed_full_bgr is not None:
            # ทำ Object Detection (Crop) โดยใช้ค่าจาก Module 1
                processed_bgr = detect_rotate_crop(processed_full_bgr, pad_x_pct=m1_pad_x, pad_y_pct=m1_pad_y, shrink=m1_shrink)
            
            # ตรวจสอบกรณี Detect ไม่เจอ
            if processed_bgr is None:
                st.error("🎯 Detection failed on processed image. Show full frame instead.")
                processed_bgr = processed_full_bgr
                
        # บันทึกผลลัพธ์
        save_path = save_dir / f"CAT_frame_{frame_num}.jpg"
        cv2.imwrite(str(save_path), processed_bgr)
        
        # Display Results
        col1, col2 = st.columns(2)
        
        # โหลดภาพต้นฉบับมาเทียบ (Crop เดิม)
        orig_crop = cv2.imread(str(crop_path))
        with col1:
            st.subheader(f"🖼️ Original Detection {frame_num}")
            if orig_crop is not None:
                st.image(cv2.cvtColor(orig_crop, cv2.COLOR_BGR2RGB), width='stretch')
            else:
                st.info("No original crop available.")
        with col2:
            method_name = st.session_state.stored_cat_method
            st.subheader(f"✨ CAT Result ({method_name})")
            st.image(cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB), width='stretch')
        with st.expander("ℹ️ Details"):
            st.write(f"**Method:** {method_name}")
            if st.session_state.stored_cat_method == "custom":
                st.write(f"**Method Details:** {st.session_state.stored_light_source} to {st.session_state.stored_light_target}")
            if method_name == "white_patch":
                st.write(f"**Percentiles:** Lower={st.session_state.stored_lower}%, Upper={st.session_state.stored_upper}%")
            st.write(f"**Settings:** {m1_pad_x}, {m1_pad_y}, {m1_shrink}")

if __name__ == "__main__":
    main()