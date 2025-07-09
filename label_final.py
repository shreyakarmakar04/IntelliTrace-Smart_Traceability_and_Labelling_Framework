import os  ## final label created including the qrcode and the data and created the final labels folder 
import cv2
import json
from pyzbar.pyzbar import decode
from PIL import Image, ImageDraw, ImageFont

# === Folder Paths ===
qr_folder = r"C:\Users\HARSHITHA\Downloads\Python\qrcodes"
output_folder = "final_labels11"  # contains list of labels here
os.makedirs(output_folder, exist_ok=True)

# === Load Fonts (Times New Roman) ===
try:
    font_path = "C:\\Windows\\Fonts\\times.ttf"
    font = ImageFont.truetype(font_path, 24)
except:
    font = ImageFont.load_default()

# === Decode QR JSON ===
def extract_qr_json(image_path):
    img = cv2.imread(image_path)
    qr_codes = decode(img)
    if qr_codes:
        try:
            return json.loads(qr_codes[0].data.decode("utf-8"))
        except:
            return None
    return None

# === Generate Compact Label ===
def generate_label(data, qr_img_path):
    if not data or "BatchID" not in data:
        print("❌ Invalid QR data format")
        return

    # Text content
    title = "CrystalClear Springs"
    subtitle = "Natural Spring Water - 500 mL"
    company = "Bottled by CrystalClear Inc., Pittsburgh, PA"
    source = "Source: Blue Ridge Springs, PA"
    mfg = f"MFG Date: {data.get('ManufacturingDate', '')}"
    exp = f"EXP Date: {data.get('ExpiryDate', '')}"
    batch = f"Batch ID: {data.get('BatchID', '')}"
    footer1 = "BPA Free | Recyclable | NSF Certified"
    footer2 = "Scan the QR code to verify authenticity."

    # Resize QR
    qr_img = Image.open(qr_img_path).convert("RGB")
    qr_size = 300
    qr_img = qr_img.resize((qr_size, qr_size))

    # Smaller label canvas
    label_width = 1000
    label_height = 500
    label = Image.new("RGB", (label_width, label_height), "white")
    draw = ImageDraw.Draw(label)

    # Text layout
    x = 30
    y = 20
    gap = 38

    draw.text((x, y), title, font=font, fill="black")  # Changed from bold_font to font
    draw.text((x, y + gap), subtitle, font=font, fill="black")
    draw.text((x, y + 2 * gap), company, font=font, fill="black")
    draw.text((x, y + 3 * gap), source, font=font, fill="black")
    draw.text((x, y + 5 * gap), mfg, font=font, fill="black")
    draw.text((x, y + 6 * gap), exp, font=font, fill="black")
    draw.text((x, y + 7 * gap), batch, font=font, fill="black")
    draw.text((x, y + 9 * gap), footer1, font=font, fill="black")
    draw.text((x, y + 10 * gap), footer2, font=font, fill="black")

    # Paste QR code
    qr_x = label_width - qr_size - 50
    qr_y = 90
    label.paste(qr_img, (qr_x, qr_y))

    # Save file
    batch_id = data.get("BatchID", "UNKNOWN")
    label.save(os.path.join(output_folder, f"Label_{batch_id}.png"))
    print(f"✅ Label generated: {batch_id}")

# === Process all QR images ===
for file in os.listdir(qr_folder):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(qr_folder, file)
        qr_data = extract_qr_json(img_path)
        if qr_data:
            generate_label(qr_data, img_path)
        else:
            print(f"❌ Failed to decode QR: {file}")
