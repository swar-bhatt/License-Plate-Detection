import cv2
import time
import re
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
import numpy as np
import torch
import easyocr
from ultralytics import YOLO


# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────

@dataclass
class Config:
    RTSP_URL: int = 0
    PLATE_MODEL: str = "best.engine"          # TensorRT engine
    CONFIDENCE: float = 0.25
    OCR_COOLDOWN_FRAMES: int = 20
    MAX_PLATES_PER_FRAME: int = 5
    TRACK_EXPIRY_FRAMES: int = 100
    IOU_THRESHOLD: float = 0.4                # for future NMS if needed
    DEVICE: int = 0                           # GPU index
    IMG_SIZE: int = 640
    USE_FP16: bool = True
    DISPLAY_WINDOW: bool = True
    FPS_SMOOTHING: int = 10                   # simple moving average


CONFIG = Config()


# ────────────────────────────────────────────────
# PLATE TRACKER
# ────────────────────────────────────────────────

class PlateTracker:
    def __init__(self):
        self.tracks: Dict[int, dict] = {}
        self.next_id: int = 1

    def update(self, detections: List[Tuple[int, int, int, int]], frame_count: int) -> List[int]:
        """Returns list of active track IDs this frame"""
        current_centers = [( (x1+x2)//2, (y1+y2)//2 ) for x1,y1,x2,y2 in detections]
        matched = set()

        for i, (cx, cy) in enumerate(current_centers):
            best_pid = None
            best_dist = float('inf')

            for pid, track in self.tracks.items():
                if pid in matched:
                    continue
                px, py = track["center"]
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if dist < 60 and dist < best_dist:  # increased threshold slightly
                    best_dist = dist
                    best_pid = pid

            if best_pid is not None:
                self.tracks[best_pid].update({
                    "center": (cx, cy),
                    "last_seen": frame_count,
                    "bbox": detections[i]
                })
                matched.add(best_pid)
            else:
                # New track
                self.tracks[self.next_id] = {
                    "center": (cx, cy),
                    "bbox": detections[i],
                    "text": "",
                    "last_ocr": -999,
                    "last_seen": frame_count
                }
                matched.add(self.next_id)
                self.next_id += 1

        # Cleanup old tracks
        to_remove = [
            pid for pid, t in self.tracks.items()
            if frame_count - t["last_seen"] > CONFIG.TRACK_EXPIRY_FRAMES
        ]
        for pid in to_remove:
            del self.tracks[pid]

        return list(matched)


# ────────────────────────────────────────────────
# IMAGE PROCESSING UTILS
# ────────────────────────────────────────────────

def preprocess_plate(image: np.ndarray) -> np.ndarray:
    """Prepare plate crop for OCR"""
    if image.size == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, d=11, sigmaColor=17, sigmaSpace=17)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # ← changed to INV (white text on black is often better)
        blockSize=31,
        C=2
    )
    return thresh


def clean_plate_text(text: str) -> str:
    """Normalize and validate plate string"""
    if not text:
        return ""
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text if 6 <= len(text) <= 10 else ""


# ────────────────────────────────────────────────
# MAIN APPLICATION
# ────────────────────────────────────────────────

class LicensePlateRecognizer:
    def __init__(self):
        print("Loading models...")
        self.detector = YOLO(CONFIG.PLATE_MODEL)
        self.ocr_reader = easyocr.Reader(['en'], gpu=True)
        self.tracker = PlateTracker()
        self.frame_count = 0
        self.fps_history = []

    def run(self):
        cap = cv2.VideoCapture(CONFIG.RTSP_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)

        if not cap.isOpened():
            print("❌ Failed to open video stream")
            return

        print("Stream opened. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Stream interrupted. Reconnecting...")
                cap.release()
                cap = cv2.VideoCapture(CONFIG.RTSP_URL)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                time.sleep(1)
                continue

            self.frame_count += 1
            start_time = time.time()

            # ── Detection ────────────────────────────────────────
            results = self.detector(
                frame,
                conf=CONFIG.CONFIDENCE,
                device=CONFIG.DEVICE,
                imgsz=CONFIG.IMG_SIZE,
                half=CONFIG.USE_FP16,
                verbose=False
            )

            # Collect bounding boxes this frame
            current_detections = []
            for r in results:
                for box in r.boxes.xyxy:
                    current_detections.append(tuple(map(int, box)))

            # ── Tracking ─────────────────────────────────────────
            active_pids = self.tracker.update(current_detections, self.frame_count)

            # ── OCR & Drawing ────────────────────────────────────
            plates_processed = 0

            for pid in active_pids:
                if plates_processed >= CONFIG.MAX_PLATES_PER_FRAME:
                    break

                track = self.tracker.tracks[pid]
                x1, y1, x2, y2 = track["bbox"]
                crop = frame[y1:y2, x1:x2]

                # OCR trigger
                should_ocr = (self.frame_count - track["last_ocr"]) > CONFIG.OCR_COOLDOWN_FRAMES
                plate_text = track["text"]

                if should_ocr and crop.size > 0:
                    processed = preprocess_plate(crop)
                    if processed is not None:
                        ocr_result = self.ocr_reader.readtext(processed, detail=0, paragraph=False)
                        if ocr_result:
                            candidate = clean_plate_text(ocr_result[0])
                            if candidate:
                                track["text"] = candidate
                                track["last_ocr"] = self.frame_count
                                plate_text = candidate

                # Draw
                color = (0, 255, 0) if plate_text else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"#{pid}"
                if plate_text:
                    label += f"  {plate_text}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                plates_processed += 1

            # ── FPS ──────────────────────────────────────────────
            elapsed = time.time() - start_time
            instant_fps = 1 / elapsed if elapsed > 0 else 0
            self.fps_history.append(instant_fps)
            if len(self.fps_history) > CONFIG.FPS_SMOOTHING:
                self.fps_history.pop(0)
            avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0

            cv2.putText(frame, f"FPS: {int(avg_fps)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)

            if CONFIG.DISPLAY_WINDOW:
                cv2.imshow("License Plate Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"Processed {self.frame_count} frames. Exiting.")


# ────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        app = LicensePlateRecognizer()
        app.run()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Unexpected error: {e}")