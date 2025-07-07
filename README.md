# IntelliTrace-Smart_Traceability_and_Labelling_Framework
An intelligent product traceability and labeling system that integrates YOLOv5-based object detection, OCR, QR/barcode decoding, and ML-based defect detection to automate quality checks and log inspection results using SQLite and Excel.

---

## 📚 Table of Contents
- [🎯 Objective](#-objective)
- [🧩 Problem Description](#-problem-description)
- [ℹ️ About the Project](#about-the-project)
- [⚙️ How It Works](#how-it-works)
- [🔁 Process Flow](#-process-flow)
- [🔍 Technical Details](#-technical-details)
- [🏗️ Architecture Diagram](#-architecture-diagram)
- [🛠️ Key Technologies and Libraries](#-key-technologies-and-libraries)
- [🧪 Dataset and Defect Detection](#-dataset-and-defect-detection)
- [📊 Results Summary](#-results-summary)
- [🖼️ Sample Output Snapshots](#-sample-output-snapshots)
- [▶️ How to Run the Project](#-how-to-run-the-project)
- [📂 Folder Structure](#-folder-structure)
- [📘 Detailed Report](#-detailed-report)
- [👨‍💻 Made by](#-made-by)

---

## 🎯 Objective
To build an AI-powered software simulation that automates product label inspection using object detection, OCR, QR/barcode decoding, and CNN-based defect detection to validate and trace manufacturing labels.

---

## 🧩 Problem Description
In high-volume manufacturing industries like electronics and medical devices, manual label verification is time-consuming and error-prone. This project addresses the challenge of:
- Automating the verification of product metadata from images
- Decoding QR and barcodes accurately
- Detecting surface defects using vision-based methods
- Logging approved/rejected products for traceability

---

## ℹ️ About the Project
This project combines real-time object detection, optical character recognition (OCR), and machine learning to create an intelligent labeling station. It:
- Uses YOLOv5 to detect and extract QR and label regions
- Extracts and validates metadata like Device ID, Batch ID, RoHS compliance
- Uses ResNet18 + PCA + OCSVM to detect visual anomalies
- Stores results in Excel and SQLite databases for traceability

---

## ⚙️ How It Works
1. **Image Input** – Static product images from dataset  
2. **Detection** – YOLOv5 detects QR and label regions  
3. **Extraction** – Pyzbar decodes QR/barcodes; Tesseract extracts text from label  
4. **Validation** – Extracted text is compared to known data (Excel/SQLite)  
5. **Defect Detection** – ResNet18 + OCSVM checks for anomalies  
6. **Output** – Decision (APPROVED / REJECTED) logged to Excel and DB  

---

## 🔁 Process Flow  
[🔍 Click to view full image](./assets/Process_Flow.png)

![Process Flow](./assets/Process_Flow.png)

---

## 🔍 Technical Details
- **Model Training**: YOLOv5s trained for 2-class detection (QR + Label)  
- **OCR & QR Decoding**: EasyOCR + Tesseract + Pyzbar  
- **CNN Anomaly Detection**: ResNet18 for feature extraction, PCA for dimensionality reduction, and One-Class SVM for classification  
- **Result Export**: Excel sheet and SQLite DB   

---

## 🏗️ Architecture Diagram  
[🔍 Click to view full image](./assets/System_Architecture.png)

![System Architecture](./assets/System_Architecture.png)

---

## 🛠️ Key Technologies and Libraries
- Python (v3.8+)
- YOLOv5 (Ultralytics)
- Tesseract OCR, EasyOCR
- Pyzbar, OpenCV
- Scikit-learn, PCA, One-Class SVM
- Streamlit (optional frontend)
- Pandas, OpenPyXL
- SQLite3

---

## 🧪 Datasets For Defect Detection(ML) & Label Detection(OpenCV)

### 🔹 Label Dataset(YOLOv5)
- 198 QR Code images + 248 Label images annotated using LabelImg
- Format: YOLOv5 `.txt` annotations

### 🔹 Defect Dataset(ResNet18)
- MVTec AD "Bottle" dataset used for ResNet training
- Binary classification: normal vs defective

---

## 📊 Results Summary

### 🔹 YOLOv5 Detection Model

| Metric     | Value |
|------------|--------|
| Precision  | 0.982  |
| Recall     | 0.967  |
| mAP@0.5    | 0.975  |
| mAP@0.5:0.95 | 0.881 |

> Trained with 198 QR images + 248 label images on YOLOv5s for 50 epochs.

---

### 🔹 ML Defect Detection Model (ResNet18 + PCA + OCSVM)

| Metric     | Value |
|------------|--------|
| Accuracy   | 98%    |
| Precision  | 100%   |
| Recall     | 95%    |
| F1-Score   | 97%    |

> Evaluated using MVTec AD Bottle dataset. Only clean images used for training. Anomalies classified via One-Class SVM.
---

## 🖼️ Sample Output Snapshots
- [Excel Summary Output](./assets/excel_output.png)
- [SQLite Log Table](./assets/sqlite_table.png)
- [Streamlit Interface](./assets/streamlit_demo.png)

---
##👨‍💻 Made by
- **Harini Mode**
- **Shreya Karmakar**
- **Tejasri Anantapalli**
