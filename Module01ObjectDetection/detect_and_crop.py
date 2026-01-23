import sys
import cv2
import os
from ultralytics import YOLOWorld

model = YOLOWorld('Module01ObjectDetection/models/yolov8s-world.pt') #yolo8 model
model.set_classes(["rectangle", "plate", "object"]) #prompt

# Function to detect objects and crop them with padding
# image_path: path to input image
# output_dir: directory to save cropped images

def detect_and_crop(image_path, padding = 0,ext = '.jpg'):
    # check image and model loading
    try:   
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit() # Exit the script if image cannot be loaded
    
    try:   
        if model is None:
            raise FileNotFoundError(f"Cannot load model")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit() # Exit the script if model cannot be loaded

    # imgsz(int) = size of image for detection
    # iou(double) = intersection over union threshold
    # max_det(int) = maximum number of detections per image
    results = model.predict(image, conf=0.0008, iou=0.45, max_det=1)
    
    #debug: create directory for cropped images
    os.makedirs("Module01ObjectDetection/cropped_objects", exist_ok=True)
    
    img_h, img_w, _ = image.shape
    output_buffers = []

    for i, result in enumerate(results):
        for j, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            nx1 = max(0, x1 - padding)
            ny1 = max(0, y1 - padding)
            nx2 = min(img_w, x2 + padding)
            ny2 = min(img_h, y2 + padding)

            cropped  = image[ny1:ny2, nx1:nx2]

            success, buf = cv2.imencode(ext, cropped)
            if success:
                output_buffers.append(buf.tobytes())

            #debug: save cropped images to files
            file_name = f"Module01ObjectDetection/cropped_objects/object_{i}_{j}_from_{os.path.basename(image_path)}"
            # Create image file
            cv2.imwrite(file_name, cropped) 
            print(f"Saved: {file_name}")
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return output_buffers

# Example usage
detect_and_crop("Module01ObjectDetection/images/input1.jpg", padding=0)