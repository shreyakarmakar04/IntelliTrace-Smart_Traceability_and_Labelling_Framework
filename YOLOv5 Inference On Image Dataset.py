# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install pytesseract pyzbar opencv-python-headless pandas ultralytics
!apt-get install -y libzbar0 tesseract-ocr


import os, cv2, numpy as np, pytesseract, re, json, time, pandas as pd, torch
from pyzbar import pyzbar
from datetime import datetime

class YOLOv5OCRDecoder:
    def __init__(self, model_path='best.pt', save_annotated_dir='annotated_output'):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        self.model.conf = 0.25
        self.model.iou = 0.45
        self.save_annotated_dir = save_annotated_dir
        os.makedirs(save_annotated_dir, exist_ok=True)

    def preprocess_for_ocr(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (gray.shape[1]*2, gray.shape[0]*2))
        gray = cv2.fastNlMeansDenoising(gray, h=30)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    def is_blurry(self, image, threshold=150.0):
        return cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < threshold

    def detect_text_regions(self, frame):
        results = self.model(frame)
        df = results.pandas().xyxy[0]
        print(df[['name', 'confidence']])  # Debug info
        return df[df['confidence'] > 0.1], results

    def draw_bounding_boxes(self, frame, detections, save_path):
        names = self.model.names
        found_classes = set()

        for _, det in detections.iterrows():
            x1, y1, x2, y2 = map(int, [det['xmin'], det['ymin'], det['xmax'], det['ymax']])
            class_id = int(det['class'])
            class_name = names.get(class_id, 'unknown')
            found_classes.add(class_name)

            color = (0, 255, 0) if class_name == 'label' else (255, 0, 0)
            label = f"{class_name} {det['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if 'qrcode' not in found_classes:
            qr_barcodes = pyzbar.decode(frame)
            for b in qr_barcodes:
                x, y, w, h = b.rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.putText(frame, 'pyzbar_qrcode', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        cv2.imwrite(save_path, frame)

    def crop_text_regions(self, frame, detections):
        return [frame[int(det['ymin']):int(det['ymax']), int(det['xmin']):int(det['xmax'])]
                for _, det in detections.iterrows()]

    def extract_manufacturing_info(self, raw_text, is_qr_data=False):
        info = {'device_id': '', 'batch_id': ''}
        flattened = raw_text.replace('\n', ' ').replace('\r', ' ').replace('"', '').replace('{', '').replace('}', '')

        if is_qr_data:
            try:
                data = json.loads(raw_text)
                info['device_id'] = str(data.get('device_id') or data.get('model') or '')
                info['batch_id'] = str(data.get('batch_id') or data.get('lot') or '')
                flattened = ', '.join([f"{k}: {v}" for k, v in data.items()])
            except:
                pass

        def valid_id(val):
            return val and 4 <= len(val) <= 12 and not val.lower().endswith(('.jpg', '.jpeg', '.png'))

        for pat in [r"Serial No[:\-]?\s*([A-Z0-9\-]{4,})", r"DeviceID[:\-]?\s*([A-Z0-9\-]{4,})", r"model[:\-]?\s*([A-Z0-9\-]{4,})"]:
            match = re.search(pat, flattened, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if valid_id(candidate):
                    info['device_id'] = candidate
                    break

        for pat in [r"Batch ID[:\-]?\s*([A-Z0-9]{4,})", r"BatchID[:\-]?\s*([A-Z0-9]{4,})", r"Lot[:\-]?\s*([A-Z0-9]{4,})"]:
            match = re.search(pat, flattened, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if valid_id(candidate):
                    info['batch_id'] = candidate
                    break

        return info, flattened

    def detect_qr_barcodes(self, frame):
        results = []
        for b in pyzbar.decode(frame):
            try:
                data = b.data.decode('utf-8')
                info, flattened = self.extract_manufacturing_info(data, is_qr_data=True)
                results.append({
                    'data': flattened,
                    'confidence': 100.0,
                    'manufacturing_info': info
                })
            except:
                results.append({
                    'data': '[UNDECODABLE]',
                    'confidence': 0.0,
                    'manufacturing_info': {'device_id': '', 'batch_id': ''}
                })
        return results

    def perform_ocr(self, imgs):
        results = []
        for img in imgs:
            processed = self.preprocess_for_ocr(img)
            text = pytesseract.image_to_string(processed, config='--psm 6').strip()
            info, flattened = self.extract_manufacturing_info(text)
            results.append({
                'text': flattened,
                'confidence': 100 if text else 0,
                'blur': self.is_blurry(img),
                'manufacturing_info': info
            })
        return results

    def determine_status(self, conf, blur, dev, batch):
        if blur: return 'Rejected'
        return 'Passed' if conf >= 30 and (dev or batch) else 'Rejected'

    def run_folder_ocr(self, folder_path, save_csv_path=None):
        results, imgs = [], os.listdir(folder_path)
        for fname in imgs:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            path = os.path.join(folder_path, fname)
            img = cv2.imread(path)
            if img is None: continue

            qr_results = self.detect_qr_barcodes(img)
            detections, _ = self.detect_text_regions(img)
            crops = self.crop_text_regions(img, detections)
            ocr_results = self.perform_ocr(crops if crops else [img])

            self.draw_bounding_boxes(img.copy(), detections, os.path.join(self.save_annotated_dir, fname))

            for res in qr_results:
                info = res['manufacturing_info']
                results.append({
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Device ID': info['device_id'],
                    'Batch ID': info['batch_id'],
                    'Status': self.determine_status(res['confidence'], False, info['device_id'], info['batch_id']),
                    'Image': path,
                    'Source': 'QR/Barcode',
                    'Raw Data': res['data']
                })

            for res in ocr_results:
                info = res['manufacturing_info']
                results.append({
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Device ID': info['device_id'],
                    'Batch ID': info['batch_id'],
                    'Status': self.determine_status(res['confidence'], res['blur'], info['device_id'], info['batch_id']),
                    'Image': path,
                    'Source': 'OCR',
                    'Raw Data': res['text']
                })

        df = pd.DataFrame(results)
        if save_csv_path:
            df.to_csv(save_csv_path, index=False, encoding='utf-8-sig')
            print(f"✅ Results saved to: {save_csv_path}")
        return df

# Paths
model_path = "/content/drive/MyDrive/best.pt"
image_folder = "/content/drive/MyDrive/label_images"
save_csv_path = "/content/drive/MyDrive/total_label.csv"
save_bboxes_folder = "/content/drive/MyDrive/final_bounded_images"

decoder = YOLOv5OCRDecoder(model_path=model_path, save_annotated_dir=save_bboxes_folder)
df = decoder.run_folder_ocr(image_folder, save_csv_path=save_csv_path)
df.head()
