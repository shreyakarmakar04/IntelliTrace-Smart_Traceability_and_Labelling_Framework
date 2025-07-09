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
- [📽️ Video Demonstration](#-)           
- [📊 Project Presentation (PPT)](#-)     
- [👨‍💻 Made by](#-made-by)

---

## 🎯 Objective
To develop an AI-powered smart product inspection and labeling system that automates the end-to-end process of label verification and defect detection. The system simulates product arrival,checks ROHS Compliance, performs object detection, OCR, barcode/QR decoding, and CNN-based defect analysis to validate key manufacturing metadata such as Device ID, Batch ID, Manufacturing Date, and RoHS Compliance. All inspection results and decisions are logged into an SQLite database and Excel report to ensure traceability, enhance quality control, and support efficient manufacturing workflows.


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
- Stores results in Excel and SQLite databases for traceability<br>

This project is a software-only simulation of an AI-powered smart traceability and labeling station for small electronic products. It automates the inspection process by integrating object detection, OCR, QR/barcode decoding, and machine learning-based defect detection. The system mimics a real-world industrial setup and handles the complete inspection lifecycle, including:<br>

1.Product Arrival Simulation<br>
2.RoHS Compliance Validation<br>
3.QR/Barcode Decoding for metadata extraction<br>
4.Label Detection & Verification using YOLOv5 and EasyOCR<br>
5.Visual Defect Detection using ResNet18 + PCA + One-Class SVM<br>
6.Approval or Rejection based on combined inspection outcomes<br>
7.Traceability Logging into Excel and SQLite databases<br>
8.It uses tools like YOLOv5, EasyOCR, OpenCV, ResNet18, PCA, One-Class SVM, SQLite, and Streamlit to create a fully functional simulation of an automated smart inspection station.

---

## ⚙️ How It Works
1.**Product Arrival:** Product enters the station via simulation or image input and RoHS Compliance & Metadata Validation done.
2. **Image Input** – Static product images from dataset  
3. **Detection** – YOLOv5 detects QR and label regions  
4. **Extraction** – Pyzbar decodes QR/barcodes; Tesseract extracts text from label  
5. **Metadata Parsed from Labels and QR Codes:**
   - 📦 **Device ID**
   - 🧪 **Batch ID**
   - 🛡️ **RoHS Compliance**
   - 📅 **Manufacturing Date**
6. **Validation** – Extracted text is compared to known data (Excel/SQLite)  
7. **Defect Detection** – ResNet18 + OCSVM checks for surface anomalies  
8. **Output** – Final status (`APPROVED` / `REJECTED`) is logged to Excel and SQLite DB
     If RoHS = "no" → REJECTED
     If OCR/QR mismatch → REJECTED
     If CNN detects defect → REJECTED
     Else → APPROVED
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

## 📽️ Video Demonstration
A complete walkthrough video showcasing the entire process—from product arrival, RoHS compliance check, label and QR code verification using YOLOv5 and OCR, ML-based defect detection, and final traceability log entry into SQLite and Excel databases. This visual explanation covers every stage of the automation flow including pass/fail decisions and rejected label handling.<br>
📎 Watch Demo Video---> [![Watch Demo](assets/demo-tumbnail.jpg)](https://drive.google.com/file/d/11Y9RNc-MASSk43raTl220d2EmFp7qu7U/view?usp=sharing)


---
## 📊 Project Presentation (PPT)
A structured PowerPoint presentation explaining the project's motivation, components, architecture, dataset usage, working methodology, and results. <br>
It includes diagrams, sample outputs, and key insights for stakeholders or reviewers.  <br>
Right-click and choose “Save Link As” to download →  
[⬇️ Download Solution Slides (PPTX)](https://github.com/shreyakarmakar04/IntelliTrace-Smart_Traceability_and_Labelling_Framework/raw/main/problem_statement_and_solution_steps_ppt.pptx)

---

👨‍💻 Made by
- **Harini Mode**
- **Shreya Karmakar**
- **Tejasri Anantapalli**
