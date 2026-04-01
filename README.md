# 🚦 Smart Road Cam

An AI-powered smart road camera system that uses computer vision and deep learning to monitor traffic in real time. Simulates a smart-city traffic surveillance solution by detecting vehicles, estimating speed, detecting accidents, reading license plates, and visualizing results through a Flask web dashboard.

---

## What It Does

| Module | File | Description |
|---|---|---|
| Vehicle detection & counting | `traffic_detection.py` | Counts vehicles crossing a virtual line; classifies traffic as low / normal / heavy every 30 seconds |
| Speed monitoring | `speed_limit.py` | Tracks vehicle centroids between frames to estimate speed; flags vehicles exceeding 30 km/h |
| Accident detection | `accident_detection.py` | Detects bounding-box overlap between vehicles; alerts when a collision is likely |
| License plate recognition | `lincense-plate.py` | Isolates plate regions via edge detection and contours; reads text with Tesseract OCR |
| Web dashboard | `app.py` + `templates/index.html` | Flask app showing live accident and traffic status |

All detection modules use **YOLOv4-Tiny** for vehicle detection (car, truck, bus) with Non-Maximum Suppression to remove duplicate boxes.

---

## Tech Stack

- **Python 3.x**
- **OpenCV** — frame capture, image processing, bounding box drawing
- **YOLOv4-Tiny** — real-time object detection (Darknet via `cv2.dnn`)
- **Tesseract OCR + pytesseract** — license plate text extraction
- **NumPy** — centroid tracking, distance calculations
- **Flask** — web dashboard
- **COCO dataset labels** — 80-class label file (`coco.names`)

---

## How It Works

### Vehicle Detection (all modules)
1. Each frame is converted to a blob and passed through YOLOv4-Tiny
2. Detections filtered by confidence > 0.5 and class in `[car, truck, bus]`
3. NMS removes overlapping boxes
4. Results are drawn back onto the frame

### Speed Estimation (`speed_limit.py`)
- Vehicle centroids are tracked frame-to-frame
- Distance travelled (pixels) × `DISTANCE_PER_PIXEL` (0.05 m/px) × FPS = speed in m/s
- Converted to km/h and checked against `SPEED_LIMIT_KMH = 30`

### Accident Detection (`accident_detection.py`)
- For every pair of detected vehicles, bounding box overlap ratio is calculated
- If overlap > 0.1 (10%), an accident is flagged on the frame

### License Plate Recognition (`lincense-plate.py`)
- Grayscale → Gaussian blur → Canny edge detection
- Contours filtered by aspect ratio and size (width > 100px, height > 30px)
- Cropped plate region passed to `pytesseract` with `--psm 8` (single word)

### Traffic Density (`traffic_detection.py`)
- Virtual horizontal line at frame midpoint counts vehicles that cross it
- Every 30 seconds: < 100 = low, 100–300 = normal, > 300 = heavy

---

## Setup

### Prerequisites

```bash
pip install opencv-python numpy flask pytesseract
```

- **Tesseract OCR** must be installed separately:
  - Windows: [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
  - Linux: `sudo apt install tesseract-ocr`
  - macOS: `brew install tesseract`

- **YOLOv4-Tiny weights** (not included in repo due to file size ~24MB):
  - Download `yolov4-tiny.weights` from [AlexeyAB/darknet releases](https://github.com/AlexeyAB/darknet/releases)
  - Place in the project root alongside `yolov4-tiny.cfg` and `coco.names`

### Video source

All scripts default to an IP camera stream. Update the URL to your source before running:

```python
# Change this line in each script:
ip_camera_url = "http://YOUR_CAMERA_IP:4747/video"

# Or use a local webcam:
cap = cv2.VideoCapture(0)

# Or a local video file:
cap = cv2.VideoCapture("traffic_sample.mp4")
```

### Run individual modules

```bash
# Traffic counting
python traffic_detection.py

# Speed monitoring
python speed_limit.py

# Accident detection
python accident_detection.py

# License plate recognition (also requires Tesseract path set)
python lincense-plate.py

# Flask web dashboard (shows combined status)
python app.py
# Then open http://127.0.0.1:5000
```

---

## Project Structure

```
smart-road-cam/
├── accident_detection.py   # Bounding box overlap collision detection
├── speed_limit.py          # Centroid-tracking speed estimation
├── traffic_detection.py    # Vehicle counting + density classification
├── lincense-plate.py       # Edge + OCR license plate reader
├── recover.py              # Alternative accident detection (proximity-based)
├── app.py                  # Flask dashboard
├── yolov4-tiny.cfg         # YOLOv4-Tiny model config
├── coco.names              # 80-class COCO label list
├── static/
│   └── css/style.css
└── templates/
    └── index.html          # Dashboard template
```

> `yolov4-tiny.weights` is not committed to the repo. Download it separately (see Setup above).

---

## Known Limitations

- Speed estimation accuracy depends on correct `DISTANCE_PER_PIXEL` calibration for your specific camera setup
- Accident detection uses simple overlap heuristics — not robust in dense traffic scenes
- License plate OCR performance degrades with low-resolution or angled plates
- All modules run on CPU; frame rate improves significantly with GPU (`DNN_TARGET_CUDA`)

---

## Author

**Divyajeetsinh Solanki** — B.Tech CSE (IoT), Manipal University Jaipur  
[github.com/devi-kisang](https://github.com/devi-kisang) · divyajeetsinhsolanki3112@gmail.com
