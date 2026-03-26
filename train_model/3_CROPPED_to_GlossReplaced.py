import os
import cv2
import numpy as np

# นำเข้า Module ของคุณ (จัด Path ของ script ให้ตรงกับตำแหน่งที่คุณรัน Streamlit นะครับ)
from Module_03_PixelSegmentation.kmeans import kmeans
from Module_03_PixelSegmentation.gloss import detect_gloss

# ================= การตั้งค่า =================
BASE_FOLDER = 'CROPPED'

# กำหนดชื่อโฟลเดอร์ Output ทั้ง 3 แบบ
OUT_DIR_2_GLOSS = 'GlossOnly'
OUT_DIR_3_REPLACED = 'GlossReplaced'

# ค่า Default สำหรับ K-means (ดึงมาจากหน้า UI)
THRESHOLD = 0.20
MAX_ITERATIONS = 1
# ==========================================

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
            
        # สร้างโฟลเดอร์ย่อยปลายทางให้ตรงกับชื่อโฟลเดอร์ต้นฉบับ
        out_path_2 = os.path.join(OUT_DIR_2_GLOSS, folder_name)
        out_path_3 = os.path.join(OUT_DIR_3_REPLACED, folder_name)
        
        os.makedirs(out_path_2, exist_ok=True)
        os.makedirs(out_path_3, exist_ok=True)
            
        # วนลูปอ่านไฟล์รูปภาพในโฟลเดอร์นั้น
        for file_name in os.listdir(folder_path):
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            file_path = os.path.join(folder_path, file_name)
            print(f"กำลังประมวลผล: {folder_name}/{file_name} ...", end=" ")
            
            # อ่านรูปภาพด้วย OpenCV (ได้มาเป็น BGR)
            img_bgr = cv2.imread(file_path)
            if img_bgr is None:
                print("❌ อ่านรูปไม่ได้")
                continue
                
            # --- 1. รัน K-means และค้นหา Gloss (ฟังก์ชันคุณรับ input เป็น path ของไฟล์) ---
            centroids, labels = kmeans(file_path, THRESHOLD, MAX_ITERATIONS)
            gloss_percent, gloss_label = detect_gloss(centroids, labels)

            # เตรียม Mask
            mask_gloss = (labels == gloss_label)
            mask_not_gloss = (labels != gloss_label)
            
            # --- 2. คำนวณค่าเฉลี่ยสี Diffuse (ส่วนที่ไม่ใช่ Gloss) เป็น BGR ---
            diffuse_pixels_bgr = img_bgr[mask_not_gloss]
            if len(diffuse_pixels_bgr) > 0:
                average_BGR_diffuse = diffuse_pixels_bgr.mean(axis=0).astype(np.uint8)
            else:
                # กรณีภาพมีแต่ Gloss ทั้งหมด ให้ fallback เป็นสีดำ
                average_BGR_diffuse = np.array([0, 0, 0], dtype=np.uint8)

            # --- 3. สร้างรูปที่ 2: บริเวณเฉพาะที่เป็น Gloss ---
            gloss_only = np.zeros_like(img_bgr)
            gloss_only[mask_gloss] = img_bgr[mask_gloss]

            # --- 4. สร้างรูปที่ 3: แทนที่ Gloss ด้วย Mean ของ Not-Gloss ---
            replaced_img = img_bgr.copy()
            if np.any(mask_not_gloss):
                replaced_img[mask_gloss] = average_BGR_diffuse
                
            # --- 5. บันทึกรูปทั้ง 3 แบบลงในโฟลเดอร์ที่จัดเตรียมไว้ ---
            # ใช้ cv2.imwrite เซฟได้เลย ไม่ต้องแปลง BGR -> RGB เพราะ OpenCV คาดหวัง BGR อยู่แล้ว
            cv2.imwrite(os.path.join(out_path_2, file_name), gloss_only)
            cv2.imwrite(os.path.join(out_path_3, file_name), replaced_img)
            
            print("✅ สำเร็จ")

    print("\n🎉 ประมวลผลและบันทึกรูปภาพทั้ง 3 โฟลเดอร์เสร็จสมบูรณ์!")