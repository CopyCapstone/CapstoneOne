import cv2
import os

def extract_one_frame_per_sec(input_folder, output_folder):
    # สร้างโฟลเดอร์ปลายทาง
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    all_files = os.listdir(input_folder)
    video_files = [f for f in all_files if f.lower().endswith(video_extensions)]
    
    print(f"พบไฟล์วิดีโอทั้งหมด {len(video_files)} ไฟล์ (โหมด 1 ภาพ/วินาที)")

    for filename in video_files:
        video_path = os.path.join(input_folder, filename)
        video_name = os.path.splitext(filename)[0] 
        
        save_folder = os.path.join(output_folder, video_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        print(f"\nกำลังดึงภาพ: {filename}")
        
        cap = cv2.VideoCapture(video_path)
        
        # 1. ดึงค่า FPS ของวิดีโอ (เช่น 24, 30, 60 เฟรมต่อวินาที)
        fps = cap.get(cv2.CAP_PROP_FPS)
        # ปัดเศษ FPS ให้เป็นจำนวนเต็ม (เผื่อวิดีโอเป็น 29.97 fps) หากอ่านไม่ได้ให้ใช้ 30 เป็นค่าเริ่มต้น
        fps_rounded = round(fps) if fps > 0 else 30 
        
        frame_count = 0
        saved_count = 0

        while True:
            # ใช้ cap.grab() เพื่อกวาดผ่านเฟรมอย่างรวดเร็ว (ไม่เสียเวลาดึงภาพจริง)
            success = cap.grab()
            
            if not success:
                break

            # 2. ตรวจสอบว่าเฟรมนี้ตรงกับวินาทีถัดไปหรือไม่ (เช่น เฟรมที่ 0, 30, 60...)
            if frame_count % fps_rounded == 0:
                # ถอดรหัสและดึงภาพออกมาเฉพาะเฟรมที่ต้องการบันทึกจริงๆ
                ret, frame = cap.retrieve()
                if ret:
                    # คำนวณว่าเป็นวินาทีที่เท่าไหร่
                    second_mark = frame_count // fps_rounded
                    
                    # ตั้งชื่อไฟล์เป็นวินาที เช่น sec_0000.jpg, sec_0001.jpg
                    frame_filename = f"{filename}_sec_{second_mark:04d}.jpg"
                    save_path = os.path.join(save_folder, frame_filename)
                    
                    cv2.imwrite(save_path, frame)
                    saved_count += 1
            
            frame_count += 1

        cap.release()
        print(f"--> บันทึกสำเร็จ {saved_count} ภาพ เก็บไว้ที่: {save_folder}")

    print("\nเสร็จสิ้นการประมวลผลวิดีโอทั้งหมด!")

def extract_all_frames(input_folder, output_folder):
    # สร้างโฟลเดอร์ปลายทาง
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    all_files = os.listdir(input_folder)
    video_files = [f for f in all_files if f.lower().endswith(video_extensions)]
    
    print(f"พบไฟล์วิดีโอทั้งหมด {len(video_files)} ไฟล์ (โหมดดึงทุกเฟรม)")

    for filename in video_files:
        video_path = os.path.join(input_folder, filename)
        video_name = os.path.splitext(filename)[0] 
        
        save_folder = os.path.join(output_folder, video_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        print(f"\nกำลังดึงภาพจาก: {filename}")
        
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        saved_count = 0

        while True:
            # ใช้ cap.read() เพื่ออ่านเฟรมทีละเฟรม
            success, frame = cap.read()
            
            if not success:
                break

            # ตั้งชื่อไฟล์เป็นเลขเฟรม เช่น frame_00000.jpg, frame_00001.jpg
            frame_filename = f"{filename}_frame_{frame_count:05d}.jpg"
            save_path = os.path.join(save_folder, frame_filename)
            
            # บันทึกภาพ
            cv2.imwrite(save_path, frame)
            saved_count += 1
            frame_count += 1

        cap.release()
        print(f"--> บันทึกสำเร็จ {saved_count} ภาพ เก็บไว้ที่: {save_folder}")

    print("\nเสร็จสิ้นการประมวลผลวิดีโอทั้งหมด!")
    

# ==========================================
# กำหนด Path ของโฟลเดอร์ตรงนี้
# ==========================================
input_directory = r"VDO"  # ใส่ path โฟลเดอร์ที่มีวิดีโอ 63 ไฟล์
output_directory = r"FRAME" # ใส่ path โฟลเดอร์ที่จะให้สร้างโฟลเดอร์ย่อย

# เรียกใช้งานฟังก์ชัน
extract_one_frame_per_sec(input_directory, output_directory)
# extract_all_frames(input_directory, output_directory)
