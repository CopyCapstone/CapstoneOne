import tensorflow as tf
import streamlit as st
import cv2
from pathlib import Path
from Module_04_ColorMeasurement.prepare_input import prepare_input
import json

# --- Configuration & Constants ---
PROJECT_ROOT = Path(__file__).parent.parent
TMP_DIR = PROJECT_ROOT / "tmp"
SETTING_FILE = TMP_DIR / 'settings.json'

# --- Initialize session state ---
def init_session_state():
    """รวมการประกาศ Session State """
    defaults = {
        "stored_predict_lab": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val  
init_session_state()

st.title("🤖 Module 4 & 6: Color Measurement and Visualization")
st.divider()
try:
    image_path = TMP_DIR / "glossReplaced_frames" / f"GlossReplaced_frame_{st.session_state.stored_frame_num}.jpg"
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        # โหลดภาพและประมวลผล
        data_raw = cv2.imread(str(image_path))
        RGB_mean = st.session_state.stored_average_RGB_diffuse 
        st.image(cv2.cvtColor(data_raw, cv2.COLOR_BGR2RGB), caption=f"GlossReplaced_frame_{st.session_state.stored_frame_num}")
        st.metric(label="Specular Replaced by Mean Diffuse", value=f"RGB: {RGB_mean}")
        # 1. โหลดโมเดล
        try:
            model = tf.keras.models.load_model('my_model.keras')
            # 2. เตรียมข้อมูล Input
            # RGB_mean มาในรูปแบบ [R, G, B]
            input_data = prepare_input(RGB_mean)
            # 3. ทำการ Predict
            prediction = model.predict(input_data)
            # 4. แสดงผลลัพธ์
            pred_L = round(float(prediction[0][0] * 100.0), 2)
            pred_a = round(float((prediction[0][1] * 240.0) - 120.0), 2)
            pred_b = round(float((prediction[0][2] * 240.0) - 120.0), 2)
            st.session_state.stored_predict_lab = [pred_L,pred_a,pred_b]
            st.metric(label=f"AI Prediction Result", value=f"L\*a\*b\*: [{pred_L} {pred_a} {pred_b}]")
        except Exception as e:
            st.error(f"Error loading or predicting: {e}") 
    with col2:
        with st.expander("ℹ️ Settings Details", expanded=True):
            st.write(f"Processing Frame: {st.session_state.stored_frame_num}")
            st.write(f"Object Detection Pad_x: {st.session_state.stored_pad_x}")
            st.write(f"Object Detection Pad_y: {st.session_state.stored_pad_y}")
            st.write(f"Object Detection Shrink: {st.session_state.stored_shrink}")
            st.write(f"Chromatic Adaptation Transform Method: {st.session_state.stored_cat_method}")
            if st.session_state.stored_cat_method == "custom":
                st.write(f"Custom CAT: {st.session_state.stored_light_source} to {st.session_state.stored_light_target}")
            if st.session_state.stored_cat_method == "white_patch":
                st.write(f"White Patch Percentile Setting: Lower={st.session_state.stored_lower}%, Upper={st.session_state.stored_upper}%")
            st.write(f"Clustering Threshold: {st.session_state.stored_threshold}")
            st.write(f"Max K-means Iterations: {st.session_state.stored_iterations}")
            st.divider()
            st.write(f"Pixel Segmentation Gloss Percentage: {st.session_state.stored_gloss_percent * 100:.2f}%")
            st.write(f"The average RGB value of specular pixels: {st.session_state.stored_average_RGB_specular}")
            st.write(f"The average RGB value of diffuse pixels: {st.session_state.stored_average_RGB_diffuse}")
            st.write(f"Prediction Result L\*a\*b\* value of diffuse pixels: {st.session_state.stored_predict_lab}")

    # Prepare the dictionary
    settings_data = {
        "pad_x": float(st.session_state.stored_pad_x),
        "pad_y": float(st.session_state.stored_pad_y),
        "shrink": float(st.session_state.stored_shrink),
        "cat_method": st.session_state.stored_cat_method,
        "clustering_threshold": float(st.session_state.stored_threshold),
        "max_kmeans_iterations": int(st.session_state.stored_iterations),
    }
    
    # Add conditional variables based on CAT method
    if st.session_state.stored_cat_method == "custom":
        settings_data["light_source"] = st.session_state.stored_light_source
        settings_data["light_target"] = st.session_state.stored_light_target
        settings_data["white_patch_lower"] = None
        settings_data["white_patch_upper"] = None

    elif st.session_state.stored_cat_method == "white_patch":
        settings_data["light_source"] = None
        settings_data["light_target"] = None
        settings_data["white_patch_lower"] = float(st.session_state.stored_lower)
        settings_data["white_patch_upper"] = float(st.session_state.stored_upper)

    # Convert dictionary to a formatted JSON string
    json_string = json.dumps(settings_data, indent=4)
    
    with st.sidebar:
        if st.button("💾 Save Settings to System"):
            try:
                with open(SETTING_FILE, 'w') as f:
                    f.write(json_string)
                st.success(f"บันทึกการตั้งค่าลงใน {SETTING_FILE} เรียบร้อยแล้ว!")                
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการบันทึก: {e}")
                
except Exception as e:
    # st.error(f"ERROR: {str(e)}")    
    st.switch_page("pages/Module3&5.py")
