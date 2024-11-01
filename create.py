import os

# Thay thế đường dẫn dưới đây bằng đường dẫn đến thư mục JPEGImages
image_folder = 'train/JPEGImages'
# Thay thế đường dẫn dưới đây bằng đường dẫn đến file trainval.txt
output_file = 'train/ImageSets/Main/train.txt'

# Tạo thư mục ImageSets/Main nếu chưa tồn tại
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Mở file để ghi
with open(output_file, 'w') as f:
    for image in os.listdir(image_folder):
        if image.endswith('.jpg') or image.endswith('.png'):  # Kiểm tra định dạng file
            f.write(image[:-4] + '\n')  # Ghi tên file mà không có đuôi

print(f'Danh sách tên tệp đã được ghi vào {output_file}')
