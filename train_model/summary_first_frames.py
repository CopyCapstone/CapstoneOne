import cv2
import os
import glob

def create_first_frames_video(input_folder, output_filename="combined_output.mp4", fps=30):
    # กำหนดนามสกุลวิดีโอที่ต้องการค้นหาในโฟลเดอร์
    valid_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv')
    video_files = []
    
    # ค้นหาไฟล์วิดีโอทั้งหมดในโฟลเดอร์
    for ext in valid_extensions:
        video_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    if not video_files:
        print(f"ไม่พบไฟล์วิดีโอในโฟลเดอร์: {input_folder}")
        return

    # เรียงลำดับไฟล์ตามชื่อ (เพื่อให้ผลลัพธ์วิดีโอเรียงลำดับถูกต้อง)
    video_files.sort()
    
    frames = []
    output_width, output_height = None, None

    print(f"กำลังดึงเฟรมแรกจากวิดีโอทั้งหมด {len(video_files)} ไฟล์...")
    
    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        ret, frame = cap.read()
        
        if ret:
            frames.append((video_file, frame))
            # กำหนดขนาดวิดีโอผลลัพธ์อ้างอิงจากเฟรมแรกที่ดึงได้สำเร็จ
            if output_width is None or output_height is None:
                output_height, output_width = frame.shape[:2]
        else:
            print(f"คำเตือน: ไม่สามารถอ่านเฟรมจากไฟล์ {os.path.basename(video_file)} ได้")
            
        cap.release()

    if not frames:
        print("ล้มเหลว: ไม่สามารถดึงเฟรมแรกจากไฟล์ใดๆ ได้เลย")
        return

    # ตั้งค่า VideoWriter สำหรับสร้างวิดีโอใหม่
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec สำหรับ mp4
    out = cv2.VideoWriter(output_filename, fourcc, fps, (output_width, output_height))

    print("กำลังสร้างวิดีโอผลลัพธ์...")
    for filename, frame in frames:
        # ปรับขนาดภาพให้เท่ากันทั้งหมด (ป้องกันกรณีวิดีโอต้นฉบับขนาดไม่เท่ากัน ซึ่งจะทำให้ VideoWriter พัง)
        # resized_frame = cv2.resize(frame, (output_width, output_height))
        
        # เขียนเฟรมเดิมซ้ำๆ จำนวนเท่ากับ fps เพื่อให้ความยาวเท่ากับ 1 วินาทีพอดี
        for _ in range(fps):
            out.write(frame)
            
    out.release()
    print(f"เสร็จสิ้น! บันทึกวิดีโอไว้ที่: {output_filename}")

# ==========================================
# วิธีใช้งาน
# ==========================================
# เปลี่ยน path ด้านล่างให้เป็นโฟลเดอร์ที่เก็บวิดีโอของคุณ
folder_path = r"VDO" 
output_path = "summary_first_frames.mp4"

create_first_frames_video(folder_path, output_filename=output_path, fps=30)