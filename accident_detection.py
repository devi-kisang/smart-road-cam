import cv2
import numpy as np
from collections import deque

# YOLO Configuration
MODEL_CONFIG = "yolov4-tiny.cfg"  # Ensure these files are in the same folder
MODEL_WEIGHTS = "yolov4-tiny.weights"
LABELS_FILE = "coco.names"

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(MODEL_CONFIG, MODEL_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class labels
with open(LABELS_FILE, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video stream (replace with your IP camera URL or webcam)
ip_camera_url = "http://10.56.129.216:4747/video"  # Update to your video source URL
cap = cv2.VideoCapture(ip_camera_url)

if not cap.isOpened():
    print("Error: Unable to open the video stream.")
    exit()

# Function to detect if two bounding boxes overlap
def boxes_overlap(box1, box2, threshold=50):
    # Calculate the intersection area between two boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Find the coordinates of the intersection box
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)

    # If there is no intersection, return False
    if ix2 < ix1 or iy2 < iy1:
        return False

    # Calculate the intersection area
    intersection_area = (ix2 - ix1) * (iy2 - iy1)
    area1 = w1 * h1
    area2 = w2 * h2

    # If the intersection area is greater than a threshold percentage of the boxes, consider it a collision
    overlap_area = intersection_area / float(min(area1, area2))
    if overlap_area > threshold:
        return True
    return False

# Function to detect accidents based on vehicle position overlap
def detect_accident(frame, boxes, class_ids):
    detected_accidents = []
    
    # Check for collisions between all pairs of vehicles
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes_overlap(boxes[i], boxes[j], threshold=0.1):  # threshold can be adjusted
                print("Accident detected between vehicles!")
                detected_accidents.append((i, j))  # Store the vehicle indices involved in the collision
    
    # If accidents are detected, add a message to the frame
    if detected_accidents:
        cv2.putText(frame, "ACCIDENT DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frame

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get YOLO layer outputs
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_layers)
    
    # Parse detections and extract bounding boxes
    boxes = []
    confidences = []
    class_ids = []
    
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5 and class_id < len(classes) and classes[class_id] in ["car", "truck", "bus"]:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Non-max suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # If indices are not empty, detect potential accidents
    if len(indices) > 0:
        frame = detect_accident(frame, [boxes[i] for i in indices.flatten()], [class_ids[i] for i in indices.flatten()])
    
    # Display the frame
    cv2.imshow("Smart Traffic Camera - Accident Detection", frame)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
