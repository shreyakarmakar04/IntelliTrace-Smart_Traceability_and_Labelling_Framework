import pandas as pd  # It creates a QRCode of the each product details and later use this qrcode in the completion of label
import json
import os 
import qrcode
from PIL import Image, ImageDraw, ImageFont

def generate_qrcodes(csv_path, output_dir='qrcodes'):
    # Load dataset from the path provided
    df = pd.read_csv(csv_path) #C:\Users\HARSHITHA\Downloads\Python\my\New Final_Traceability_Updated.csv

    # Filter rows where RoHSCompliance == "yes"
    df = df[df['RoHSCompliance'].str.lower() == 'yes']

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Group by BatchID
    batch_groups = df.groupby('BatchID')

    for batch_id, group in batch_groups:
        record = group.iloc[0]

        data_to_encode = {
            'DeviceID': record['DeviceID'],
            'BatchID': record['BatchID'],
            'ManufacturingDate': record['ManufacturingDate'],
            'SerialNumber': record['SerialNumber'],
            'Location': record['Location'],
            'RoHSCompliance': record['RoHSCompliance']
        }
        data_str = json.dumps(data_to_encode, separators=(',', ':'))

        # Generate QR code
        qr = qrcode.QRCode(
            version=2,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4,
        )
        qr.add_data(data_str)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white").convert('RGB')

        img_w, img_h = img.size

        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        new_img_h = img_h + 30
        new_img = Image.new("RGB", (img_w, new_img_h), "white")
        new_img.paste(img, (0, 0))

        draw = ImageDraw.Draw(new_img)
        serial_text = str(record['SerialNumber'])
        bbox = draw.textbbox((0, 0), serial_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = (img_w - text_w) // 2
        text_y = img_h + 5
        draw.text((text_x, text_y), serial_text, fill="black", font=font)

        final_path = os.path.join(output_dir, f"qrcode_batch_{batch_id}.png")
        new_img.save(final_path)

        print(f"Generated QR code for BatchID {batch_id} (RoHSCompliance=yes): {final_path}")

if __name__ == "__main__":
    csv_path = input("Please enter the full path to your CSV file: ").strip('"')
    generate_qrcodes(csv_path)
