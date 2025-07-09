# STEP 1: Setup
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt
!pip install easyocr pyzbar openpyxl scikit-learn 

# STEP 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# STEP 3: Dataset Split
import os, shutil
from sklearn.model_selection import train_test_split

base_path = "/content/drive/MyDrive/label dataset"
images_dir = os.path.join(base_path, "images")
labels_dir = os.path.join(base_path, "labels")

splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(f"/content/datasets/images/{split}", exist_ok=True)
    os.makedirs(f"/content/datasets/labels/{split}", exist_ok=True)

def move_split(files, split):
    for file in files:
        base = os.path.splitext(file)[0]
        img_src = os.path.join(images_dir, file)
        lbl_src = os.path.join(labels_dir, base + '.txt')
        if os.path.exists(lbl_src):
            shutil.copy(img_src, f"/content/datasets/images/{split}/{file}")
            shutil.copy(lbl_src, f"/content/datasets/labels/{split}/{base}.txt")

image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
train_val, test = train_test_split(image_files, test_size=0.1, random_state=42)
train, val = train_test_split(train_val, test_size=0.2222, random_state=42)

move_split(train, "train")
move_split(val, "val")
move_split(test, "test")

# STEP 4: Create data.yaml
yaml_path = "/content/yolov5/data.yaml"
with open(yaml_path, 'w') as f:
    f.write("""
path: /content/datasets
train: images/train
val: images/val
test: images/test
nc: 2
names: ['qrcode', 'label']
""")

# STEP 5: Train Model
!python train.py --img 640 --batch 16 --epochs 50 --data /content/yolov5/data.yaml --weights yolov5s.pt --name qr_label_detection

# STEP 6: Print Model Architecture
import torch
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/qr_label_detection/weights/best.pt')
print(model.model)

# STEP 7: Export to ONNX and TFLite
!python export.py --weights runs/train/qr_label_detection/weights/best.pt --include onnx tflite --img 640

# STEP 8: Copy to Drive
!cp runs/train/qr_label_detection/weights/best.pt /content/drive/MyDrive/best.pt
!cp runs/train/qr_label_detection/weights/best.onnx /content/drive/MyDrive/best.onnx
!cp runs/train/qr_label_detection/weights/best.tflite /content/drive/MyDrive/best.tflite
