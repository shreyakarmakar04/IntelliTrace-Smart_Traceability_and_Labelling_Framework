import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import threading
from collections import deque


class YOLOv5QRDecoderLive:
    def __init__(self, model_path='best.pt'):
        """
        Initialize YOLOv5 model for live QR code detection

        Args:
            model_path: Path to YOLOv5 model weights
        """
        print("Loading YOLOv5 model...")
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

        # Set confidence threshold
        self.model.conf = 0.25  # NMS confidence threshold
        self.model.iou = 0.45  # NMS IoU threshold

        # Initialize QR detector
        self.qr_detector = cv2.QRCodeDetector()

        # For storing recent QR codes to avoid spam
        self.recent_qr_codes = deque(maxlen=10)
        self.last_detection_time = {}

        print("Model loaded successfully!")

    def detect_qr_codes(self, frame):
        """
        Detect QR codes in frame using YOLOv5

        Args:
            frame: OpenCV frame from webcam

        Returns:
            List of detected QR code bounding boxes
        """
        # Run inference
        results = self.model(frame)

        # Extract detection results
        detections = results.pandas().xyxy[0]  # pandas DataFrame

        # Filter for QR codes - adjust this based on your model's class names
        if len(detections) > 0:
            # If your model has specific QR class names, filter by them
            # For a general object detection model, you might want to return all detections
            # and let the QR decoder validate them
            qr_detections = detections[detections['confidence'] > 0.3]  # Adjust confidence as needed
        else:
            qr_detections = detections

        return qr_detections

    def crop_qr_regions(self, frame, detections):
        """
        Crop QR code regions from frame based on detections

        Args:
            frame: OpenCV frame
            detections: DataFrame of YOLOv5 detections

        Returns:
            List of cropped QR code images and their coordinates
        """
        cropped_qrs = []
        coordinates = []

        for idx, detection in detections.iterrows():
            # Extract bounding box coordinates
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(
                detection['ymax'])

            # Add some padding to ensure we get the full QR code
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)

            # Crop QR code region
            cropped_qr = frame[y1:y2, x1:x2]
            cropped_qrs.append(cropped_qr)
            coordinates.append((x1, y1, x2, y2))

        return cropped_qrs, coordinates

    def decode_qr_codes(self, qr_images):
        """
        Decode QR codes using OpenCV QR detector

        Args:
            qr_images: List of cropped QR code images

        Returns:
            List of decoded QR code information
        """
        decoded_info = []

        for i, qr_img in enumerate(qr_images):
            try:
                # Convert to grayscale if needed
                if len(qr_img.shape) == 3:
                    gray = cv2.cvtColor(qr_img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = qr_img

                # Detect and decode QR code
                data, bbox, _ = self.qr_detector.detectAndDecode(gray)

                if data:
                    qr_data = {
                        'qr_index': i,
                        'type': 'QRCODE',
                        'data': data,
                        'bbox': bbox,
                        'timestamp': time.time()
                    }
                    decoded_info.append(qr_data)
                else:
                    # Try with image enhancement
                    enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,
                                                     2)
                    data, bbox, _ = self.qr_detector.detectAndDecode(enhanced)

                    if data:
                        qr_data = {
                            'qr_index': i,
                            'type': 'QRCODE',
                            'data': data,
                            'bbox': bbox,
                            'timestamp': time.time()
                        }
                        decoded_info.append(qr_data)

            except Exception as e:
                print(f"Error decoding QR code {i}: {str(e)}")

        return decoded_info

    def is_new_qr_code(self, qr_data, cooldown_seconds=2):
        """
        Check if QR code is new or recently detected (to avoid spam)

        Args:
            qr_data: QR code data string
            cooldown_seconds: Seconds to wait before showing same QR again

        Returns:
            True if QR code should be displayed/processed
        """
        current_time = time.time()

        if qr_data in self.last_detection_time:
            if current_time - self.last_detection_time[qr_data] < cooldown_seconds:
                return False

        self.last_detection_time[qr_data] = current_time
        return True

    def draw_results(self, frame, detections, decoded_info, coordinates):
        """
        Draw bounding boxes and QR code information on frame

        Args:
            frame: OpenCV frame
            detections: YOLOv5 detections
            decoded_info: Decoded QR information
            coordinates: Bounding box coordinates

        Returns:
            Frame with annotations
        """
        annotated_frame = frame.copy()

        # Draw YOLOv5 detection boxes
        for idx, detection in detections.iterrows():
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(
                detection['ymax'])

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw confidence score
            conf_text = f"Conf: {detection['confidence']:.2f}"
            cv2.putText(annotated_frame, conf_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw decoded QR information
        y_offset = 30
        for i, info in enumerate(decoded_info):
            if self.is_new_qr_code(info['data']):
                # Print to console for new QR codes
                print(f"\n{'=' * 50}")
                print(f"NEW QR CODE DETECTED:")
                print(f"Data: {info['data']}")
                print(f"Type: {info['type']}")
                print(f"Time: {time.strftime('%H:%M:%S', time.localtime(info['timestamp']))}")
                print(f"{'=' * 50}")

            # Display on frame
            qr_text = f"QR {i + 1}: {info['data'][:50]}{'...' if len(info['data']) > 50 else ''}"
            cv2.putText(annotated_frame, qr_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 30

            # Draw QR-specific bounding box if available
            if i < len(coordinates):
                x1, y1, x2, y2 = coordinates[i]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated_frame, "QR", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return annotated_frame

    def run_webcam(self, camera_index=0, window_name="YOLOv5 QR Code Detection - Live"):
        """
        Run live QR code detection from webcam

        Args:
            camera_index: Camera index (0 for default camera)
            window_name: OpenCV window name
        """
        print(f"Starting webcam (camera index: {camera_index})...")
        print("Press 'q' to quit, 's' to save current frame, 'c' to clear recent QR history")

        # Initialize webcam
        cap = cv2.VideoCapture(camera_index)

        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return

        # Performance tracking
        fps_counter = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                try:
                    # Step 1: Detect QR codes with YOLOv5
                    detections = self.detect_qr_codes(frame)

                    decoded_info = []
                    coordinates = []

                    if len(detections) > 0:
                        # Step 2: Crop QR code regions
                        cropped_qrs, coordinates = self.crop_qr_regions(frame, detections)

                        # Step 3: Decode QR codes
                        if cropped_qrs:
                            decoded_info = self.decode_qr_codes(cropped_qrs)

                    # Step 4: Draw results on frame
                    annotated_frame = self.draw_results(frame, detections, decoded_info, coordinates)

                except Exception as e:
                    print(f"Error processing frame: {e}")
                    annotated_frame = frame

                # Calculate and display FPS
                fps_counter += 1
                if fps_counter % 30 == 0:  # Update FPS every 30 frames
                    elapsed_time = time.time() - start_time
                    fps = fps_counter / elapsed_time
                    print(f"FPS: {fps:.1f}")

                # Add FPS to frame
                fps_text = f"FPS: {fps_counter / (time.time() - start_time):.1f}" if time.time() - start_time > 0 else "FPS: --"
                cv2.putText(annotated_frame, fps_text, (annotated_frame.shape[1] - 100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Add instructions
                cv2.putText(annotated_frame, "Press 'q' to quit, 's' to save, 'c' to clear history",
                            (10, annotated_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Display frame
                cv2.imshow(window_name, annotated_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"qr_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"Frame saved as {filename}")
                elif key == ord('c'):
                    # Clear recent QR codes history
                    self.last_detection_time.clear()
                    print("QR code detection history cleared")

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Webcam stopped")

    def run_webcam_with_logging(self, camera_index=0, log_file="qr_detections.txt"):
        """
        Run webcam with QR code logging to file

        Args:
            camera_index: Camera index
            log_file: File to log QR detections
        """
        self.log_file = log_file
        print(f"QR detections will be logged to: {log_file}")

        # Create/clear log file
        with open(log_file, 'w') as f:
            f.write(f"QR Code Detection Log - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

        self.run_webcam(camera_index)


# Main execution
def main():
    """
    Main function to run live QR code detection
    """
    # Initialize the QR decoder with your trained model
    model_path = 'best.pt'  # Change this to your model path

    try:
        qr_decoder = YOLOv5QRDecoderLive(model_path)

        # Run webcam detection
        qr_decoder.run_webcam(camera_index=0)  # Use camera_index=1 for external camera

    except Exception as e:
        print(f"Error initializing QR decoder: {e}")
        print("Make sure your model file exists and all dependencies are installed")


def main_with_logging():
    """
    Main function with QR code logging
    """
    model_path = 'best.pt'

    try:
        qr_decoder = YOLOv5QRDecoderLive(model_path)
        qr_decoder.run_webcam_with_logging(camera_index=0, log_file="qr_log.txt")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run basic webcam detection
    main()

    # Uncomment to run with logging
    # main_with_logging()
