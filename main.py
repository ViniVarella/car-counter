# Importation
from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
import os
import numpy as np
import torch
from sort import *

# Initialization and variable naming
MODEL_PATH = "yolov8l.pt"
VIDEO_PATH = r"assets/traffic_cam.mp4"
INFERENCE_SIZE = 320
PROCESS_EVERY_N_FRAMES = 2
CONFIDENCE_THRESHOLD = 0.35
USE_MASK = False
MASK_PATH = "assets/mask.png"
SHOW_MASK_PREVIEW = False
DEVICE = 0 if torch.cuda.is_available() else "cpu"

COUNT_LINES = [
    {
        "name": "Up",
        "x": 175,
        "y": 500,
        "angle": 2,
        "length": 400,
    },
    {
        "name": "Down",
        "x": 715,
        "y": 500,
        "angle": 1,
        "length": 350,
    },
]

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo nao encontrado: {MODEL_PATH}")
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video nao encontrado: {VIDEO_PATH}")

model = YOLO(MODEL_PATH)
vid = cv.VideoCapture(VIDEO_PATH)
if not vid.isOpened():
    raise RuntimeError(f"Nao foi possivel abrir o video: {VIDEO_PATH}")

device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Rodando inferencia em: {device_name}")

mask = None
if USE_MASK:
    mask = cv.imread(MASK_PATH)
    if mask is None:
        raise FileNotFoundError(f"Mascara nao encontrada: {MASK_PATH}")

class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

tracker = Sort(max_age = 22, min_hits = 3, iou_threshold = 0.3)
line_counts = [[] for _ in COUNT_LINES]
line_flash_frames = [0 for _ in COUNT_LINES]
last_detections = np.empty((0, 5))
frame_index = 0

# Setting up video writer properties (for saving the output result)
width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = vid.get(cv.CAP_PROP_FPS)
video_writer = cv.VideoWriter(("result.mp4"), cv.VideoWriter_fourcc("m", "p", "4", "v"),
                              fps, (width, height))
FLASH_DURATION_FRAMES = max(6, int((fps or 30) * 0.2))


def scale_line(points, frame_width, frame_height, base_width=1280, base_height=720):
    return [
        int(points[0] * frame_width / base_width),
        int(points[1] * frame_height / base_height),
        int(points[2] * frame_width / base_width),
        int(points[3] * frame_height / base_height),
    ]


def build_line_from_angle(start_point, angle_degrees, length_pixels):
    start_x, start_y = start_point
    angle_radians = math.radians(angle_degrees)
    end_x = int(round(start_x + math.cos(angle_radians) * length_pixels))
    # In screen coordinates, positive angles should visually go upward.
    end_y = int(round(start_y - math.sin(angle_radians) * length_pixels))
    return [start_x, start_y, end_x, end_y]


count_lines = [
    build_line_from_angle((line["x"], line["y"]), line["angle"], line["length"])
    for line in COUNT_LINES
]


def draw_hud(frame, counts):
    frame_height, frame_width = frame.shape[:2]
    panel_width = min(max(260, int(frame_width * 0.58)), frame_width - 20)
    panel_height = min(max(92, int(frame_height * 0.2)), frame_height - 20)
    top_left = (10, 10)
    bottom_right = (10 + panel_width, 10 + panel_height)

    overlay = frame.copy()
    cv.rectangle(overlay, top_left, bottom_right, (25, 25, 25), cv.FILLED)
    cv.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv.rectangle(frame, top_left, bottom_right, (255, 255, 255), 2)

    title_scale = max(0.42, frame_width / 980)
    value_scale = max(0.62, frame_width / 620)
    title_thickness = max(1, frame_width // 280)
    value_thickness = max(1, frame_width // 220)
    total = sum(len(ids) for ids in counts)
    label_y = top_left[1] + 24
    value_y = top_left[1] + 90
    col_width = panel_width // 3

    cv.putText(frame, "Total", (top_left[0] + 12, label_y), cv.FONT_HERSHEY_SIMPLEX, title_scale, (230, 230, 230), title_thickness)
    cv.putText(frame, str(total), (top_left[0] + 12, value_y), cv.FONT_HERSHEY_SIMPLEX, value_scale, (255, 90, 220), value_thickness)

    if len(counts) > 0:
        up_x = top_left[0] + col_width + 12
        cv.putText(frame, "Up", (up_x, label_y), cv.FONT_HERSHEY_SIMPLEX, title_scale, (230, 230, 230), title_thickness)
        cv.putText(frame, str(len(counts[0])), (up_x, value_y), cv.FONT_HERSHEY_SIMPLEX, value_scale, (80, 255, 120), value_thickness)

    if len(counts) > 1:
        down_x = top_left[0] + (col_width * 2) + 12
        cv.putText(frame, "Down", (down_x, label_y), cv.FONT_HERSHEY_SIMPLEX, title_scale, (230, 230, 230), title_thickness)
        cv.putText(frame, str(len(counts[1])), (down_x, value_y), cv.FONT_HERSHEY_SIMPLEX, value_scale, (80, 200, 255), value_thickness)


def point_to_segment_distance(px, py, line):
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)

    projection = ((px - x1) * dx + (py - y1) * dy) / float(dx * dx + dy * dy)
    projection = max(0.0, min(1.0, projection))
    nearest_x = x1 + projection * dx
    nearest_y = y1 + projection * dy
    return math.hypot(px - nearest_x, py - nearest_y)


def crossed_lane_bar(cx, cy, line, tolerance):
    x1, y1, x2, y2 = line
    if not (min(x1, x2) - tolerance <= cx <= max(x1, x2) + tolerance):
        return False
    if not (min(y1, y2) - tolerance <= cy <= max(y1, y2) + tolerance):
        return False
    return point_to_segment_distance(cx, cy, line) <= tolerance

while True:
    ref, frame = vid.read()
    if not ref or frame is None:
        break
    frame_index += 1

    if USE_MASK and mask is not None:
        if mask.shape[0] != frame.shape[0] or mask.shape[1] != frame.shape[1]:
            frame_mask = cv.resize(mask, (frame.shape[1], frame.shape[0]))
        else:
            frame_mask = mask
        frame_region = cv.bitwise_and(frame, frame_mask)
    else:
        frame_region = frame

    detections = np.empty((0, 5))

    if frame_index % PROCESS_EVERY_N_FRAMES == 0:
        result = model(
            frame_region,
            imgsz=INFERENCE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
            device=DEVICE,
        )

        for r in result:
            boxes = r.boxes
            for box in boxes:
                # Bounding boxes
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                #Detection confidence
                conf = math.floor(box.conf[0]*100)/100

                # Class names
                cls = int(box.cls[0])
                vehicle_names = class_names[cls]

                if vehicle_names == "car" or vehicle_names == "truck" or vehicle_names == "bus"\
                    or vehicle_names == "motorbike":
                    current_detection = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, current_detection))
        last_detections = detections
    else:
        detections = last_detections

    # Tracking codes
    tracker_updates = tracker.update(detections)
    # Tracking lines
    for index, count_line in enumerate(count_lines):
        count_line_color = (0, 255, 0) if line_flash_frames[index] > 0 else (0, 0, 255)
        cv.line(frame, (count_line[0], count_line[1]), (count_line[2], count_line[3]), count_line_color, thickness = 3)

    # Geting bounding boxes points and vehicle ID
    for update in tracker_updates:
        x1, y1, x2, y2, id = update
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = (x2-x1), (y2-y1)

        # Getting tracking marker
        cx, cy = (x1+w//2), (y1+h//2)
        cv.circle(frame, (cx, cy), 5, (255, 0, 255), cv.FILLED)
        lane_tolerance = max(12, min(width, height) // 24)

        for index, count_line in enumerate(count_lines):
            if crossed_lane_bar(cx, cy, count_line, lane_tolerance):
                if line_counts[index].count(id) == 0:
                    line_counts[index].append(id)
                line_flash_frames[index] = FLASH_DURATION_FRAMES

        # Adding rectangles and texts
        cvzone.cornerRect(frame, (x1, y1, w, h), l=5, colorR=(255, 0, 255), rt=1)
        cvzone.putTextRect(frame, f'{id}', (x1, y1), scale=1, thickness=2)

    draw_hud(frame, line_counts)

    for index, frames_remaining in enumerate(line_flash_frames):
        if frames_remaining > 0:
            line_flash_frames[index] -= 1

    if SHOW_MASK_PREVIEW:
        cv.imshow("vid", frame_region)
    else:
        cv.imshow("vid", frame)

    # Saving the video frame output
    video_writer.write(frame)

    if cv.waitKey(1) & 0xFF == 27:
        break

# Closing down everything
vid.release()
cv.destroyAllWindows()
video_writer.release()
