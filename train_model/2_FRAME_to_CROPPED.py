import sys
import cv2
import os
import numpy as np
from ultralytics import YOLO

# base directory ของ module นี้
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yoloTag.pt")
model = YOLO(MODEL_PATH)

# ใส่ภาพที่ต้องการตรวจจับและครอบวัตถุ
# image_path = "path/to/your/image.jpg"
# pad_y_pct = 0.1 # ปรับเปอร์เซ็นต์ของ padding ในแนวตั้ง
# pad_x_pct = 0.1 # ปรับเปอร์เซ็นต์ของ padding ในแนวนอน
# shrink = 0.1 # ปรับเปอร์เซ็นต์ของการหดขอบหลังจากหมุนให้ตรง
def detect_rotate_crop(image, pad_x_pct=-0.1, pad_y_pct=-0.05, shrink=0.2):

    results = model.predict(image, conf=0.25, max_det=1, verbose=False)

    if len(results) == 0:
        return None

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return None

    box = result.boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    img_h, img_w = image.shape[:2]

    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = int(box_w * pad_x_pct)
    pad_y = int(box_h * pad_y_pct)

    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(img_w, x2 + pad_x)
    ny2 = min(img_h, y2 + pad_y)

    cropped = image[ny1:ny2, nx1:nx2]

    # straighten
    blur_bgr = cv2.GaussianBlur(cropped, (5, 5), 0)
    edges = cv2.Canny(blur_bgr, 5, 60)

    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]

    _, mask_color = cv2.threshold(s, 25, 255, cv2.THRESH_BINARY)

    combined_mask = cv2.bitwise_or(edges, mask_color)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        rotated = cropped
    else:
        cnt = max(contours, key=cv2.contourArea)

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect

        if angle < -45:
            angle += 90

        h_img, w_img = cropped.shape[:2]

        M = cv2.getRotationMatrix2D((w_img // 2, h_img // 2), angle, 1.0)

        rotated = cv2.warpAffine(
            cropped,
            M,
            (w_img, h_img),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

    # final crop
    h, w = rotated.shape[:2]

    dx = int(w * shrink)
    dy = int(h * shrink)

    final = rotated[dy:h-dy, dx:w-dx]

    return final


# เข้าไป folder FRAME ใน นั้นมี 63 folder เข้าไปใช้ detect_rotate_crop ทุกภาพในทุก folder แล้วบันทึกผลลัพธ์ลงใน folder OUTPUT
def process_all_folders(input_base_folder="FRAME", output_base_folder="OUTPUT"):
    # สร้างโฟลเดอร์ OUTPUT หลักถ้ายังไม่มี
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)

    # เช็คว่ามีโฟลเดอร์ FRAME อยู่จริงหรือไม่
    if not os.path.exists(input_base_folder):
        print(f"ไม่พบโฟลเดอร์ {input_base_folder} ครับ กรุณาตรวจสอบ Path")
        return

    # ดึงรายชื่อโฟลเดอร์ย่อยทั้งหมดใน FRAME
    subfolders = [f for f in os.listdir(input_base_folder) if os.path.isdir(os.path.join(input_base_folder, f))]
    print(f"พบโฟลเดอร์ย่อยทั้งหมด {len(subfolders)} โฟลเดอร์")

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    total_processed = 0
    total_saved = 0

    # วนลูปทีละโฟลเดอร์ย่อย (63 โฟลเดอร์)
    for folder_name in subfolders:
        input_folder_path = os.path.join(input_base_folder, folder_name)
        output_folder_path = os.path.join(output_base_folder, folder_name)

        # สร้างโฟลเดอร์ย่อยใน OUTPUT ให้ชื่อตรงกัน
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        # ดึงรายชื่อไฟล์ภาพในโฟลเดอร์ย่อยนี้
        image_files = [f for f in os.listdir(input_folder_path) if f.lower().endswith(valid_extensions)]
        
        if not image_files:
            continue

        print(f"\nกำลังประมวลผลโฟลเดอร์: {folder_name} ({len(image_files)} ภาพ)")

        # วนลูปอ่านและประมวลผลภาพทีละไฟล์
        for img_name in image_files:
            img_path = os.path.join(input_folder_path, img_name)
            save_path = os.path.join(output_folder_path, img_name)

            # อ่านภาพ
            image = cv2.imread(img_path)
            if image is None:
                print(f"  [Error] ไม่สามารถอ่านไฟล์ภาพได้: {img_path}")
                continue
            
            total_processed += 1

            try:
                # เรียกใช้ฟังก์ชันที่คุณเขียนไว้
                result_img = detect_rotate_crop(image)

                # ถ้าตรวจพบวัตถุและ Crop สำเร็จ ให้บันทึกรูป
                if result_img is not None:
                    cv2.imwrite(save_path, result_img)
                    total_saved += 1
                else:
                    print(f"  [Skip] ไม่พบวัตถุ หรือประมวลผลไม่ได้ในภาพ: {img_name}")
            except Exception as e:
                print(f"  [Error] เกิดข้อผิดพลาดกับภาพ {img_name}: {e}")

    print("\n=========================================")
    print(f"เสร็จสิ้นภารกิจ!")
    print(f"ประมวลผลไปทั้งหมด: {total_processed} ภาพ")
    print(f"บันทึกสำเร็จ: {total_saved} ภาพ")
    print("=========================================")

# ==========================================
# ส่วนเรียกใช้งาน
# ==========================================
if __name__ == "__main__":
    # ใส่ชื่อโฟลเดอร์ต้นทางและปลายทาง (สามารถปรับเป็น Absolute Path ได้ถ้าวางไฟล์ไว้คนละที่)
    INPUT_DIR = "FRAME" 
    OUTPUT_DIR = "CROPPED"
    
    process_all_folders(INPUT_DIR, OUTPUT_DIR)