import pandas as pd # this the final_one with all the details into the rejecyed_log.csv and approved are sent for the next phase 
import os
from PIL import Image
import winsound
import easyocr
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURATION ===
csv_path = r'C:\Users\HARSHITHA\Downloads\Python\my\Final_Traceability_With_Images.csv'
product_image_folder = r'C:\Users\HARSHITHA\Downloads\Python\my\bottle_50'
approved_sound = r'C:\Users\HARSHITHA\Downloads\Python\my\Product Approved Sen.wav'
rejected_sound = r'C:\Users\HARSHITHA\Downloads\Python\my\Product Rejected Sen.wav'
rejected_log_path = r'C:\Users\HARSHITHA\Downloads\Python\my\Rejected_Log.csv'

# === LOAD CSV ===
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# === OCR Reader (optional for later use) ===
reader = easyocr.Reader(['en'])

# === FUNCTION: Show image using matplotlib ===
def show_image(img_path, title="Image"):
    try:
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f" Failed to open image '{img_path}': {e}")

# === MAIN LOOP ===
while True:
    batch_id = input("\n🔷 Enter Batch ID to verify (or type 'exit' to quit): ").strip()
    if batch_id.lower() == "exit":
        print("✅ Exiting system. Goodbye!")
        break

    matching_rows = df[df['BatchID'].astype(str).str.strip() == batch_id]

    if matching_rows.empty:
        winsound.PlaySound("SystemHand", winsound.SND_ALIAS)
        print(f"❌ Batch ID '{batch_id}' not found in CSV.")
        continue

    valid_row = matching_rows[matching_rows['Image_Filename'].notna()].head(1)
    if valid_row.empty:
        valid_row = matching_rows.head(1)

    row = valid_row.iloc[0]

    device_id = str(row['DeviceID']).strip()
    manufacture_date = str(row['ManufacturingDate']).strip()
    serial_no = str(row['SerialNumber']).strip()
    rohs = str(row['RoHSCompliance']).strip().lower()
    image_filename = str(row.get('Image_Filename', '')).strip()

    print(f"""\n✅ Product Found:
Device ID         : {device_id}
Batch ID          : {batch_id}
Manufacturing Date: {manufacture_date}
Serial Number     : {serial_no}
RoHS Compliance   : {rohs.upper()}""")

    # === SHOW PRODUCT IMAGE ONLY ===
    if image_filename and image_filename.lower() != 'nan':
        product_img_path = os.path.join(product_image_folder, image_filename)
        if os.path.exists(product_img_path):
            show_image(product_img_path, title=f"Product Image: {batch_id}")
        else:
            print(f" Product image '{image_filename}' not found in folder.")
    else:
        print(f" No image filename specified for Batch ID: {batch_id}")

    # === PLAY SOUND & LOG REJECTED IF NEEDED ===
    if rohs == 'yes':
        print("✅ ROHS Passed → Send to Conveyor A for OCR")
        winsound.PlaySound(approved_sound, winsound.SND_FILENAME)
    else:
        print("❌ ROHS Failed → Send to Conveyor B for Review")
        winsound.PlaySound(rejected_sound, winsound.SND_FILENAME)

        # --- LOG REJECTED ROW TO EXCEL/CSV IF NOT ALREADY LOGGED ---
        rejected_row = row.copy()
        rejected_row['ValidationStatus'] = 'Rejected'

        if os.path.exists(rejected_log_path):
            existing_log = pd.read_csv(rejected_log_path)
            existing_log['BatchID'] = existing_log['BatchID'].astype(str).str.strip()
            if batch_id not in existing_log['BatchID'].values:
                updated_log = pd.concat([existing_log, pd.DataFrame([rejected_row])], ignore_index=True)
                updated_log.to_csv(rejected_log_path, index=False)
                print(f" Rejected entry logged to: {rejected_log_path}")
            else:
                print("Rejected Batch ID already logged earlier. Skipping duplicate. ROHS Failed so Rejected and logged for final review.")
        else:
            updated_log = pd.DataFrame([rejected_row])
            updated_log.to_csv(rejected_log_path, index=False)
            print(f" Rejected entry logged to: {rejected_log_path}")
