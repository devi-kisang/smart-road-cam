import cv2
import numpy as np
from collections import deque
import time

# YOLO Configuration
MODEL_CONFIG = "yolov4-tiny.cfg"  # Ensure this file is in the same folder
MODEL_WEIGHTS = "yolov4-tiny.weights"  # Ensure this file is in the same folder
LABELS_FILE = "coco.names"  # File containing class labels (e.g., car, truck)

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(MODEL_CONFIG, MODEL_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class labels
with open(LABELS_FILE, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video stream from IP camera or webcam
# Replace this with your live stream URL or use 0 for webcam
ip_camera_url = "http://10.56.129.216:4747/video"  # Example for IP camera
cap = cv2.VideoCapture(ip_camera_url)

if not cap.isOpened():
    print(f"Error: Unable to open live video stream at {ip_camera_url}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Speed threshold in km/h (converted from pixels per second)
SPEED_LIMIT_KMH = 30  # Speed limit in km/h

# Vehicle tracking initialization
vehicle_count = 0
line_position = int(frame_height / 2)  # Define a line for counting vehicles
centroids_previous = {}  # Track previous centroids for speed calculation

# Calibration: Define distance per pixel (in meters)
DISTANCE_PER_PIXEL = 0.05  # Example: 5 cm per pixel, adjust according to your setup

# Timer for resetting traffic count
last_reset_time = time.time()

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: Failed to grab frame. Exiting...")
        break

    # Prepare frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get YOLO layer outputs
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    detections = net.forward(output_layers)

    # Parse detections
    height, width, _ = frame.shape
    boxes = []
    confidences = []
    class_ids = []
    centroids = []  # Track the centroid positions of vehicles

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter detections by confidence and class
            if confidence > 0.5 and classes[class_id] in ["car", "truck", "bus"]:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                centroids.append((center_x, center_y))

    # Apply Non-Max Suppression to filter overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Check the correct format of indices
    if len(indices) > 0:
        indices = indices.flatten()  # Flatten if indices is a tuple

    # Calculate speed and detect overspeeding
    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        label = str(classes[class_ids[i]])
        confidence = confidences[i]

        # Draw bounding boxes and labels
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {int(confidence * 100)}%", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Get the centroid position of the vehicle
        center_x = x + w // 2
        center_y = y + h // 2
        centroid = (center_x, center_y)

        # Calculate speed if the vehicle was detected in the previous frame
        if centroid in centroids_previous:
            prev_centroid = centroids_previous[centroid]
            # Calculate the distance traveled in pixels
            distance_pixels = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
            # Convert pixels to meters
            distance_meters = distance_pixels * DISTANCE_PER_PIXEL
            # Calculate speed in meters per second
            speed_mps = distance_meters * fps
            # Convert speed to km/h
            speed_kmh = (speed_mps * 3600) / 1000

            # Check if the vehicle is overspeeding
            if speed_kmh > SPEED_LIMIT_KMH:
                cv2.putText(frame, "OVERSPEEDING!", (x, y - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                print(f"Overspeeding detected! Vehicle speed: {speed_kmh:.2f} km/h")

        # Store the current centroid for speed calculation in the next frame
        centroids_previous[centroid] = centroid

    # Draw a line for vehicle counting
    cv2.line(frame, (0, line_position), (frame_width, line_position), (0, 0, 255), 2)

    # Display the frame with detected vehicles
    cv2.imshow("Traffic Detection - Overspeeding", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
