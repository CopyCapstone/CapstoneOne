import streamlit as st

# กำหนดหน้าต่างๆ โดยตั้งให้ Module1.py เป็นหน้าแรกด้วยคำสั่ง default=True
Module1 = st.Page("pages/Module1.py", default=True)
Module2 = st.Page("pages/Module2.py")
Module3 = st.Page("pages/Module3&5.py")
Module4 = st.Page("pages/Module4&6.py")

# นำหน้าเพจทั้งหมดไปใส่ใน navigation
pg = st.navigation([Module1,Module2,Module3,Module4])

# สั่งรันการนำทาง
pg.run()