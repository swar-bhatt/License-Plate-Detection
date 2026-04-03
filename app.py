import cv2
import time
import re
import csv
import os
from datetime import datetime
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
    OCR_COOLDOWN_SECONDS: float = 0.5   
    MAX_PLATES_PER_FRAME: int = 5
    TRACK_EXPIRY_FRAMES: int = 100
    IOU_THRESHOLD: float = 0.4                # for future NMS if needed
    PLATE_MIN_LEN: int = 6      # ← ADD
    PLATE_MAX_LEN: int = 10     # ← ADD
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
                x1, y1, x2, y2 = detections[i]
                bbox_diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if dist < bbox_diag * 0.5 and dist < best_dist:  # increased threshold slightly
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
                    "text_votes": {},
                    "last_ocr_time": 0.0,
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
# PLATE LOGGER
# ────────────────────────────────────────────────

class PlateLogger:
    def __init__(self, filepath: str = "plate_log.csv"):
        self.filepath = filepath
        self.records: Dict[str, dict] = {}  # keyed by plate text

        # Create CSV with headers if it doesn't exist
        if not os.path.exists(filepath):
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "plate_number",
                    "first_seen",
                    "last_seen",
                    "read_count",
                    "best_confidence",
                    "all_readings"
                ])
        print(f"📋 Plate log: {os.path.abspath(filepath)}")

    def log(self, plate_text: str, confidence: float):
        """Call this every time a plate is successfully read."""
        if not plate_text:
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if plate_text not in self.records:
            # Brand new plate
            self.records[plate_text] = {
                "plate_number":   plate_text,
                "first_seen":     now,
                "last_seen":      now,
                "read_count":     1,
                "best_confidence": round(confidence, 3),
                "all_readings":   [f"{now}(conf:{confidence:.2f})"]
            }
            print(f"🆕 New plate detected: {plate_text} at {now}")
        else:
            # Existing plate — update record
            rec = self.records[plate_text]
            rec["last_seen"]  = now
            rec["read_count"] += 1
            rec["all_readings"].append(f"{now}(conf:{confidence:.2f})")
            if confidence > rec["best_confidence"]:
                rec["best_confidence"] = round(confidence, 3)
            print(f"🔄 Updated plate: {plate_text} | reads: {rec['read_count']} | best conf: {rec['best_confidence']}")

        self._write_csv()

    def _write_csv(self):
        """Rewrite the full CSV with current records."""
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "plate_number", "first_seen", "last_seen",
                "read_count", "best_confidence", "all_readings"
            ])
            for rec in self.records.values():
                writer.writerow([
                    rec["plate_number"],
                    rec["first_seen"],
                    rec["last_seen"],
                    rec["read_count"],
                    rec["best_confidence"],
                    " | ".join(rec["all_readings"])
                ])

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

    thresh_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 31, 2)
    thresh_norm = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 31, 2)
    return [thresh_inv, thresh_norm]   # return both, OCR loop will handle both


def clean_plate_text(text: str) -> str:
    """Normalize and validate plate string"""
    if not text:
        return ""
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text if CONFIG.PLATE_MIN_LEN <= len(text) <= CONFIG.PLATE_MAX_LEN else ""

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
        self.logger = PlateLogger("plate_log.csv")
        
    def run(self):
        cap = cv2.VideoCapture(CONFIG.RTSP_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)

        if not cap.isOpened():
            print("❌ Failed to open video stream")
            return

        print("Stream opened. Press 'q' to quit.")
        retries = 0
        MAX_RETRIES = 5
        while True:
            ret, frame = cap.read()
            if ret:
                retries = 0
            if not ret:
                retries += 1
                print(f"Stream interrupted. Retry {retries}/{MAX_RETRIES}...")
                cap.release()
                if retries > MAX_RETRIES:
                    print("Stream unavailable. Exiting.")
                    break
                time.sleep(min(2 ** retries, 30))
                cap = cv2.VideoCapture(CONFIG.RTSP_URL)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
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
                for box in r.boxes.xyxy.cpu().numpy():
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
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)   
                x2, y2 = min(w, x2), min(h, y2)
                crop = frame[y1:y2, x1:x2]

                # OCR trigger
                should_ocr = (time.time() - track.get("last_ocr_time", 0)) > CONFIG.OCR_COOLDOWN_SECONDS
                plate_text = track["text"]

                if should_ocr and crop.size > 0:
                    best_candidate, best_conf = "", 0.0
                    preprocessed = preprocess_plate(crop)
                    for img in [crop] + (preprocessed if isinstance(preprocessed, list) else [preprocessed]):
                        if img is None or img.size == 0:
                            continue
                        ocr_result = self.ocr_reader.readtext(img, detail=1, paragraph=False,
                                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                        if ocr_result:
                            text, conf = ocr_result[0][1], ocr_result[0][2]
                            candidate = clean_plate_text(text)
                            if candidate and conf > best_conf:
                                best_candidate, best_conf = candidate, conf
                    if best_candidate:
                        votes = track.setdefault("text_votes", {})
                        votes[best_candidate] = votes.get(best_candidate, 0) + 1
                        best_voted = max(votes, key=votes.get)
                        if votes[best_voted] >= 2:
                            track["text"] = best_voted
                            plate_text = best_voted
                            self.logger.log(best_voted, best_conf)
                        track["last_ocr_time"] = time.time()
                for img in [crop] + (preprocessed if isinstance(preprocessed, list) else [preprocessed]):
                    if img is None or img.size == 0:
                        continue
                    ocr_result = self.ocr_reader.readtext(img, detail=1, paragraph=False,
                                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    if ocr_result:
                        text, conf = ocr_result[0][1], ocr_result[0][2]
                        candidate = clean_plate_text(text)
                        if candidate and conf > best_conf:
                            best_candidate, best_conf = candidate, conf
                if best_candidate:
                    votes = track.setdefault("text_votes", {})
                    votes[best_candidate] = votes.get(best_candidate, 0) + 1
                    best_voted = max(votes, key=votes.get)
                    if votes[best_voted] >= 2:          # confirmed after 2 consistent reads
                        track["text"] = best_voted
                        plate_text = best_voted
                    track["last_ocr_time"] = time.time()

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
