import streamlit as st

st.title("🎥 Upload Video")

st.write(st.session_state.video_path)
st.write(st.session_state.select_frame_num)
