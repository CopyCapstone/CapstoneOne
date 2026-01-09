import cv2
import os
from ultralytics import YOLOWorld

model = YOLOWorld('Module01ObjectDetection/models/yolov8s-world.pt') #yolo8 model

model.set_classes(["rectangle", "plate", "object"]) #promt

image_path = "Module01ObjectDetection/images/input.jpg" #for testing
image = cv2.imread(image_path)
results = model.predict(image, conf= 0.005)

output_dir = "Module01ObjectDetection/cropped_objects"
os.makedirs(output_dir, exist_ok=True)

padding = 0 #no padding
img_h, img_w, _ = image.shape

for i, result in enumerate(results):
    for j, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        nx1 = max(0, x1 - padding)
        ny1 = max(0, y1 - padding)
        nx2 = min(img_w, x2 + padding)
        ny2 = min(img_h, y2 + padding)
        
        cropped_img = image[ny1:ny2, nx1:nx2]
        
        file_name = f"{output_dir}/object_{i}_{j}.jpg" #for testing
        cv2.imwrite(file_name, cropped_img)
        print(f"Saved: {file_name}")

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

#cv2.imshow("Detection Result", image) #for show areas
cv2.waitKey(0)
cv2.destroyAllWindows()