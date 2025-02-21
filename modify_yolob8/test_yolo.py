from ultralytics import YOLO
import os

def main():
  model_path = "./ultralytics/ultralytics/cfg/models/v8/yolov8.yaml"
  data_path = "./ultralytics/ultralytics/cfg/datasets/coco8.yaml"
  # Khởi tạo model YOLOv8n từ file cấu hình
  model = YOLO(model_path)

  # Tạo thư mục runs/detect nếu chưa tồn tại
  os.makedirs("runs/detect", exist_ok=True)

  # Huấn luyện mô hình
  model.train(
      # Đường dẫn file .yaml cho dataset
      data=data_path,
      epochs=10,       # Số vòng lặp huấn luyện
      batch=16,        # Batch size
      lr0=1e-3,         # Learning rate (tăng lên để học nhanh hơn)
      momentum=0.9,    # Tăng momentum giúp mô hình hội tụ tốt hơn
      optimizer="adam",  # Lựa chọn optimizer (có thể là "SGD" hoặc "adam")
      weight_decay=0.0005  # Giảm overfitting
  )


if __name__ == "__main__":
  main()
