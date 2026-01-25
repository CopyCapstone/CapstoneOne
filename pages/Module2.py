import streamlit as st
import cv2
import os
from colour.colorimetry import CCS_ILLUMINANTS
from Module_02_ChromaticAdaptationTransform.cat import chromatic_adaptation_transform
from Module_02_ChromaticAdaptationTransform.custom_CAT import custom_CAT 
from skimage.color import rgb2xyz, xyz2rgb
import numpy as np

st.title("Module 2: CAT")
# ---------------------------------------------------------
# 1. State Management
# ---------------------------------------------------------
# สร้างตัวแปร "ถาวร" (Persistent) สำหรับเก็บค่า Setting
if "stored_cat_method" not in st.session_state:
    st.session_state.stored_cat_method = "grey_world"
if "stored_lower" not in st.session_state:
    st.session_state.stored_lower = 95
if "stored_upper" not in st.session_state:
    st.session_state.stored_upper = 99
if "stored_light_source" not in st.session_state:
    st.session_state.stored_light_source = 'D65'
if "stored_light_target" not in st.session_state:
    st.session_state.stored_light_target = 'D65'

# ฟังก์ชัน Callback: เมื่อ Widget เปลี่ยนค่า -> ให้บันทึกลงตัวแปรถาวรทันที
def update_settings():
    st.session_state.stored_cat_method = st.session_state.widget_method
    # เช็คก่อนว่ามี key ของ slider หรือไม่ (กัน error กรณีเลือก grey_world)
    if "widget_lower" in st.session_state:
        st.session_state.stored_lower = st.session_state.widget_lower
    if "widget_upper" in st.session_state:
        st.session_state.stored_upper = st.session_state.widget_upper
    if "widget_source" in st.session_state:
        st.session_state.stored_light_source = st.session_state.widget_source
    if "widget_target" in st.session_state:
        st.session_state.stored_light_target = st.session_state.widget_target

# ---------------------------------------------------------
# 2. Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("Settings")
# หา Index เริ่มต้นของ Selectbox จากค่าที่บันทึกไว้
method_options = ["grey_world", "white_patch","custom"]
try:
    default_index = method_options.index(st.session_state.stored_cat_method)
except ValueError:
    default_index = 0

# Selectbox
st.sidebar.selectbox(
    "CAT Method", 
    method_options, 
    index=default_index,
    key="widget_method",
    on_change=update_settings
)

# Slider (จะแสดงก็ต่อเมื่อเลือก white_patch)
if st.session_state.stored_cat_method == "white_patch":
    st.sidebar.slider(
        "Lower Percentile", 0, 100, 
        value=st.session_state.stored_lower,
        key="widget_lower", 
        on_change=update_settings
    )
    st.sidebar.slider(
        "Upper Percentile", 0, 100, 
        value=st.session_state.stored_upper,
        key="widget_upper", 
        on_change=update_settings
    )

# Slider (จะแสดงก็ต่อเมื่อเลือก custom)
if st.session_state.stored_cat_method == "custom":
    ILLUMINANTS_xy = CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
    illuminant_list = list(ILLUMINANTS_xy.keys())

    # 1. เลือก Source Illuminant (แสงต้นฉบับของรูป)
    try:
        idx_src = illuminant_list.index(st.session_state.stored_light_source)
    except ValueError: idx_src = 0
    st.sidebar.selectbox(
        "Source Illuminant (แสงต้นทาง)", 
        illuminant_list, 
        index=idx_src,
        key="widget_source", 
        on_change=update_settings
    )

    # 2. เลือก Target Illuminant (แสงที่อยากได้)
    try:
        idx_tgt = illuminant_list.index(st.session_state.stored_light_target)
    except ValueError: idx_tgt = illuminant_list.index('D65') # default D65
    st.sidebar.selectbox(
        "Target Illuminant (แสงปลายทาง)", 
        illuminant_list, 
        index=idx_tgt,
        key="widget_target",
        on_change=update_settings
    )


# ---------------------------------------------------------
# 3. Image Processing Logic
# ---------------------------------------------------------
frame_num = st.session_state.stored_frame_num
st.subheader(f"Processing Frame: {frame_num}")
image_dir = "tmp/cropped_objects"
save_dir = "tmp/CAT_objects"
os.makedirs("tmp/CAT_objects", exist_ok=True)
image_filename = f"cropped_frame_{frame_num}.jpg"
image_path = os.path.join(image_dir, image_filename)

if os.path.exists(image_path):
    current_method = st.session_state.stored_cat_method
    current_lower = st.session_state.stored_lower
    current_upper = st.session_state.stored_upper
    
    # โหลดรูปต้นฉบับเพื่อแสดงผลเทียบ
    img_bgr = cv2.imread(image_path)
    if current_method == "custom":
        # 1. เตรียมค่า xy ของแสง
        xy_src = ILLUMINANTS_xy[st.session_state.stored_light_source]
        xy_tgt = ILLUMINANTS_xy[st.session_state.stored_light_target]

        # 2. แปลง BGR -> RGB -> XYZ (High Precision using skimage)
        # ขั้นแรกแปลง BGR (OpenCV) เป็น RGB ก่อน
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # ใช้ skimage แปลง RGB เป็น XYZ
        # หมายเหตุ: skimage จะปรับค่าจาก uint8 (0-255) เป็น float (0-1) ให้โดยอัตโนมัติ
        # และจัดการเรื่อง sRGB Gamma linearization ให้ด้วย ทำให้แม่นยำกว่า cv2
        img_xyz = rgb2xyz(img_rgb)
        
        # 3. เรียกใช้ฟังก์ชัน custom_CAT
        processed_xyz = custom_CAT(img_xyz, xy_src, xy_tgt, method='Von Kries')
        
        # 4. แปลง XYZ กลับเป็น RGB -> BGR
        # ใช้ skimage แปลงกลับ (Output เป็น float)
        processed_rgb_float = xyz2rgb(processed_xyz)
        
        # สำคัญ! ต้อง Clip ค่าให้อยู่ในช่วง [0, 1] 
        # เพราะการเปลี่ยน White Point อาจทำให้สีบางสีหลุดออกนอก Gamut (ค่าติดลบ หรือเกิน 1)
        processed_rgb_float = np.clip(processed_rgb_float, 0, 1)
        
        # แปลงจาก float (0-1) กลับเป็น uint8 (0-255) เพื่อแสดงผล
        processed_rgb_uint8 = (processed_rgb_float * 255).astype(np.uint8)
        
        # แปลงกลับเป็น BGR เพื่อให้ Flow ของโปรแกรมข้างล่างทำงานต่อได้ถูกต้อง
        processed_bgr = cv2.cvtColor(processed_rgb_uint8, cv2.COLOR_RGB2BGR)
    else:
        # ใช้ image_path ส่งไปให้ฟังก์ชัน
        processed_bgr = chromatic_adaptation_transform(
            image_path, 
            current_method, 
            [current_lower, current_upper]
        )

    # Save image
    save_path = os.path.join(save_dir, f"CAT_frame_{frame_num}.jpg")
    cv2.imwrite(save_path, processed_bgr)
    
    if img_bgr is not None and processed_bgr is not None:
        original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Original Crop")
            st.image(original_rgb, width='stretch')
        with col2:
            if current_method == "custom":
                st.markdown(f"### Result ({st.session_state.stored_light_source} to {st.session_state.stored_light_target})")
            else:
                st.markdown(f"### Result ({current_method})")
            st.image(processed_rgb, width='stretch')
    else:
        st.error("Error: Could not decode image.")
        
else:
    st.warning(f"File not found: `{image_path}`")