import streamlit as st
import cv2
import os

st.title("Module 3: Light Detection")
st.subheader(f"{st.session_state.stored_cat_method} Processing Frame: {st.session_state.stored_frame_num}")
if st.session_state.stored_cat_method == "custom":
    st.subheader(f"{st.session_state.stored_light_source} to {st.session_state.stored_light_target}")
if st.session_state.stored_cat_method == "white_patch":
    st.subheader(f"**Percentiles:** Lower={st.session_state.stored_lower}%, Upper={st.session_state.stored_upper}%")
