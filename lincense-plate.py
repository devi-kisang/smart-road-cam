import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Set up pytesseract path (adjust if necessary)
# Ensure Tesseract is installed: https://github.com/tesseract-ocr/tesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'  # Change for your OS

# Define the IP Camera URL (replace with your phone's streaming URL)
IP_CAMERA_URL = "http://10.56.129.216:4747/video"  # Replace with your IP camera stream URL

# Function for processing frames
def detect_license_plate(frame):
    """
    Process the frame to detect license plates using OpenCV and pytesseract.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
    edges = cv2.Canny(blurred, 30, 200)  # Edge detection
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:  # Approximate to rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 100 and h > 30:  # Filtering based on size
                license_plate = frame[y:y + h, x:x + w]
                text = pytesseract.image_to_string(license_plate, config='--psm 8')
                return text, (x, y, w, h)
    return None, None

# Start video capture
cap = cv2.VideoCapture(IP_CAMERA_URL)

if not cap.isOpened():
    print("Error: Unable to connect to the IP Camera.")
    exit()

print("Connected to IP Camera. Press any key in the matplotlib window to quit.")

# Set up matplotlib for displaying frames
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Detect license plate
    detected_text, bbox = detect_license_plate(frame)
    if detected_text:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
        cv2.putText(frame, detected_text.strip(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        print(f"Detected License Plate: {detected_text.strip()} at {datetime.now()}")

    # Convert BGR (OpenCV) to RGB (matplotlib)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax.imshow(frame_rgb)
    plt.pause(0.001)  # Pause to update the frame
    ax.clear()

    if plt.waitforbuttonpress(0.001):  # Break on key press
        break

# Close matplotlib window and release resources
plt.close(fig)
cap.release()
