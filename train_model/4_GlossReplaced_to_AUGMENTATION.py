import os
import cv2
import numpy as np
import pandas as pd
import random

# ================= การตั้งค่า (ตั้งค่าตรงนี้ได้เลย) =================
BASE_FOLDER = 'GlossReplaced'           # ชื่อโฟลเดอร์หลักที่มี 63 โฟลเดอร์ย่อยอยู่ข้างใน
OUTPUT_CSV = 'rgb_augmented.csv' # ชื่อไฟล์ผลลัพธ์ที่จะเซฟ
NUM_CROPS = 5                  # จำนวนครั้งที่จะสุ่ม Crop ต่อ 1 รูป (Augmentation multiplier)
CROP_PERCENT = 0.20             # สัดส่วนขนาด Crop เป็น 10% ของภาพ (0.10)
# ==========================================================

data_records = []

# ตรวจสอบว่าโฟลเดอร์หลักมีอยู่จริง
if not os.path.exists(BASE_FOLDER):
    print(f"ไม่พบโฟลเดอร์: {BASE_FOLDER} โปรดตรวจสอบ Path อีกครั้ง")
else:
    # วนลูปเข้าไปในแต่ละโฟลเดอร์ย่อย (63 โฟลเดอร์)
    for folder_name in os.listdir(BASE_FOLDER):
        folder_path = os.path.join(BASE_FOLDER, folder_name)
        
        # เช็คให้แน่ใจว่าเป็นโฟลเดอร์ ไม่ใช่ไฟล์
        if not os.path.isdir(folder_path):
            continue
            
        # วนลูปอ่านไฟล์รูปภาพในโฟลเดอร์นั้น
        for file_name in os.listdir(folder_path):
            # กรองเอาเฉพาะไฟล์รูปภาพ
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            file_path = os.path.join(folder_path, file_name)
            
            # อ่านรูปภาพด้วย OpenCV (ระวัง: OpenCV อ่านมาเป็น BGR)
            img_bgr = cv2.imread(file_path)
            if img_bgr is None:
                print(f"อ่านรูปไม่ได้: {file_path}")
                continue
                
            # แปลง BGR เป็น RGB ให้ถูกต้อง
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = img_rgb.shape
            
            # ---------------------------------------------------------
            # ส่วนที่แก้ไข: คำนวณขนาดที่ตัดให้เป็น 10% ของกว้างและสูงของภาพนี้
            # ใช้ max(1, ...) เพื่อป้องกันไม่ให้ขนาดกลายเป็น 0 พิกเซล
            # ---------------------------------------------------------
            crop_w = max(1, int(w * CROP_PERCENT))
            crop_h = max(1, int(h * CROP_PERCENT))

            # ---------------------------------------------------------
            # วนลูปทำ Random Crop
            # ---------------------------------------------------------
            for crop_idx in range(NUM_CROPS):
                # ตรวจสอบว่าภาพเล็กกว่าขนาดที่จะ Crop หรือไม่
                if w <= crop_w or h <= crop_h:
                    crop_img = img_rgb.copy() # ถ้ารูปเล็กไป ให้ใช้รูปเต็มเลย
                else:
                    # สุ่มพิกัดมุมซ้ายบน (x, y) ไม่ให้เกินขอบเขตของภาพ
                    x = random.randint(0, w - crop_w)
                    y = random.randint(0, h - crop_h)
                    
                    # ตัดภาพ (Crop) ด้วยเทคนิค NumPy Slicing [แกนy, แกนx]
                    crop_img = img_rgb[y:y+crop_h, x:x+crop_w]

                # คำนวณค่าเฉลี่ยสีของภาพที่ Crop ได้
                rgb_mean = np.mean(crop_img, axis=(0, 1))
                r_mean, g_mean, b_mean = rgb_mean[0], rgb_mean[1], rgb_mean[2]
            # ---------------------------------------------------------

                # save crop_img ไปที่โฟลเดอร์ใหม่ (ถ้าต้องการเก็บรูปที่ตัดไว้ด้วย)
                crop_folder = os.path.join('AUGMENTATION', folder_name)
                os.makedirs(crop_folder, exist_ok=True)
                # ตั้งชื่อไฟล์ใหม่โดยใส่เลข crop index เข้าไป
                crop_file_name = f"{os.path.splitext(file_name)[0]}_crop{crop_idx + 1:02d}.png"
                crop_file_path = os.path.join(crop_folder, crop_file_name)
                
                # บันทึกเป็น BGR เพื่อให้ OpenCV เขียนไฟล์ได้สีถูกต้อง
                cv2.imwrite(crop_file_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)) 
                
                # เก็บข้อมูลลงลิสต์
                data_records.append({
                    'ID': folder_name,          # ชื่อโฟลเดอร์ (ซึ่งมักจะตรงกับ ID เช่น M-136_1)
                    'File_Name': file_name,     # ชื่อไฟล์รูป (ต้นฉบับ)
                    'Crop_Index': crop_idx + 1, # รหัส Crop (1 ถึง NUM_CROPS)
                    'R_mean': round(r_mean, 4), # ปัดทศนิยม 4 ตำแหน่ง
                    'G_mean': round(g_mean, 4),
                    'B_mean': round(b_mean, 4)
                })

    # นำข้อมูลทั้งหมดมาสร้างเป็นตาราง (DataFrame) และเซฟเป็น CSV
    if len(data_records) > 0:
        df = pd.DataFrame(data_records)
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        print(f"✅ ประมวลผลเสร็จสิ้น! บันทึกไฟล์ไปที่: {OUTPUT_CSV}")
        print(f"📊 ได้ข้อมูลทั้งหมด {len(df)} แถว จาก {len(df['ID'].unique())} หมวดหมู่")
    else:
        print("❌ ไม่พบข้อมูลรูปภาพให้ประมวลผล")