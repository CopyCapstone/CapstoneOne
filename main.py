import streamlit as st

st.set_page_config(page_title="Capstone Project", layout="wide")

# กำหนดหน้าต่างๆ โดยตั้งให้ Module1.py เป็นหน้าแรกด้วยคำสั่ง default=True
Module1 = st.Page("pages/Module1.py", title = "1) Object Detection", default=True, icon = "🔍")
Module2 = st.Page("pages/Module2.py", title = "2) Chromatic Adaptation Transform", icon = "😼")
Module3 = st.Page("pages/Module3&5.py", title = "3) Pixel Segmentation and Glossiness Measurement", icon = "📚")
Module4 = st.Page("pages/Module4&6.py", title = "4) Color Measurement and Visualization", icon = "🤖")

# นำหน้าเพจทั้งหมดไปใส่ใน navigation
pg = st.navigation([Module1,Module2,Module3,Module4])

# สั่งรันการนำทาง
pg.run()