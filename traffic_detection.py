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

# Initialize video stream from IP camera
ip_camera_url = "http://10.56.129.216:4747/video"
cap = cv2.VideoCapture(ip_camera_url)

if not cap.isOpened():
    print(f"Error: Unable to open IP camera stream at {ip_camera_url}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Traffic counter and vehicle tracking initialization
vehicle_count = 0
line_position = int(frame_height / 2)  # Define a line for counting vehicles

# Track vehicles using a deque for storing centroids
tracking_history = deque(maxlen=30)  # Store the last 30 frames of tracking data
centroids_previous = set()  # Store centroids from the previous frame for tracking

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

    # Count vehicles crossing the line
    for centroid in centroids:
        if centroid not in centroids_previous:
            if centroid[1] > line_position:  # Check if centroid is below the line
                vehicle_count += 1
        centroids_previous = set(centroids)

    # Draw bounding boxes and labels
    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        label = str(classes[class_ids[i]])
        confidence = confidences[i]

        # Draw bounding boxes and labels
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {int(confidence * 100)}%", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw a line for vehicle counting
    cv2.line(frame, (0, line_position), (frame_width, line_position), (0, 0, 255), 2)

    # Display traffic count on the frame
    cv2.putText(frame, f"Traffic Count: {vehicle_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Check if it's time to reset the count (every 30 seconds)
    current_time = time.time()
    if current_time - last_reset_time >= 30:
        # Reset vehicle count every 30 seconds
        last_reset_time = current_time
        # Output traffic status based on count
        if vehicle_count < 100:
            print("No traffic")
        elif 100 <= vehicle_count <= 300:
            print("Normal traffic")
        else:
            print("Too much traffic")
        
        # Reset traffic count after checking
        vehicle_count = 0

    # Display the frame
    cv2.imshow("Traffic Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
