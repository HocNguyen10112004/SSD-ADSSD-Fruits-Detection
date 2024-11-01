import torch
import cv2
import numpy as np
from torchvision.transforms import functional as F
from model.ssd import SSD
import yaml
import matplotlib.pyplot as plt

# Tải cấu hình từ file YAML
with open('config/voc.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Cấu hình dữ liệu và mô hình
dataset_config = config['dataset_params']
train_config = config['train_params']

# Khởi tạo mô hình SSD và tải trọng số
model = SSD(config=config['model_params'], num_classes=dataset_config['num_classes'])
model.load_state_dict(torch.load('voc/ssd_voc2007.pth', map_location=torch.device('cpu')))
model.eval()

# Hàm để xử lý ảnh và thực hiện dự đoán
def predict(image_path):
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return

    height, width, _ = image.shape

    # Chuyển đổi ảnh sang RGB
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Chuyển đổi ảnh thành tensor và thêm batch dimension
    image_tensor = F.to_tensor(image).unsqueeze(0)

    # Dự đoán
    with torch.no_grad():
        loss, predictions = model(image_tensor)

    # Lấy bounding boxes và scores
    boxes = predictions[0]['boxes'].numpy()  # Giả sử box đã được chuẩn hóa [0, 1]
    scores = predictions[0]['scores'].numpy()
    labels = predictions[0]['labels'].numpy()

    # Chỉ lấy những dự đoán có độ tin cậy cao (thí dụ: > 0.5)
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.5:
            # Resize bounding box về kích thước của ảnh gốc
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'Label: {label}, Score: {score:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Hiển thị ảnh bằng matplotlib
    plt.imshow(image)
    plt.axis('off')  # Tắt trục
    plt.show()

# Gọi hàm dự đoán với đường dẫn tới ảnh
predict('test/JPEGImages/11a50eb1f43d8727_jpg.rf.c7753892562ceaae31599738ea21cd57.jpg')
