import cv2
import time
import re
from dataclasses import dataclass
from typing import Tuple, Dict, List
import numpy as np
import torch
import easyocr
from ultralytics import YOLO


# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────

@dataclass
class Config:
    RTSP_URL: int | str = 0                   # 0 = webcam, or "rtsp://..."
    PLATE_MODEL: str = "best.engine"          # Your TensorRT engine file
    CONFIDENCE: float = 0.25
    OCR_COOLDOWN_FRAMES: int = 30             # Less frequent OCR → lower CPU/GPU load
    MAX_PLATES_PER_FRAME: int = 5
    TRACK_EXPIRY_FRAMES: int = 120
    DEVICE: int = 0                           # GPU index for YOLO
    IMG_SIZE: int = 640
    USE_FP16: bool = True
    DISPLAY_WINDOW: bool = True
    FPS_SMOOTHING: int = 10
    MIN_PLATE_AREA: int = 1200                # Skip very small/distant plates
    OCR_CONF_THRESHOLD: float = 0.45          # Minimum OCR confidence
    OCR_ALLOWLIST: str = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # Restrict charset


CONFIG = Config()


# ────────────────────────────────────────────────
# PLATE TRACKER (simple centroid-based)
# ────────────────────────────────────────────────

class PlateTracker:
    def __init__(self):
        self.tracks: Dict[int, dict] = {}
        self.next_id: int = 1

    def update(self, detections: List[Tuple[int, int, int, int]], frame_count: int) -> List[int]:
        """Returns list of active track IDs this frame"""
        current_centers = [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2 in detections]
        matched = set()

        for i, (cx, cy) in enumerate(current_centers):
            best_pid = None
            best_dist = float('inf')

            for pid, track in self.tracks.items():
                if pid in matched:
                    continue
                px, py = track["center"]
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if dist < 70 and dist < best_dist:  # generous distance threshold
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

        # Remove stale tracks
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

def preprocess_plate(image: np.ndarray) -> np.ndarray | None:
    """Improved preprocessing for license plate OCR"""
    if image.size == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=15, sigmaSpace=15)

    # Light sharpening
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    gray = cv2.filter2D(gray, -1, kernel)

    # Binary threshold - most plates have dark text on light bg
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=2
    )

    return thresh


def clean_plate_text(text: str) -> str:
    """Clean and validate extracted plate text"""
    if not text:
        return ""
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    # Most plates are 5–10 chars; adjust range as needed for your country
    return text if 5 <= len(text) <= 10 else ""


# ────────────────────────────────────────────────
# MAIN APPLICATION
# ────────────────────────────────────────────────

class LicensePlateRecognizer:
    def __init__(self):
        print("Loading models...")
        # Explicitly set task='detect' to silence warning
        self.detector = YOLO(CONFIG.PLATE_MODEL, task='detect')
        self.ocr_reader = easyocr.Reader(
            ['en'],
            gpu=False,                  # OCR on CPU → frees GPU for faster YOLO inference
            model_storage_directory=None  # default
        )
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
                time.sleep(1.5)
                continue

            self.frame_count += 1
            start_time = time.time()

            # ── YOLO Detection ───────────────────────────────────
            results = self.detector(
                frame,
                conf=CONFIG.CONFIDENCE,
                device=CONFIG.DEVICE,
                imgsz=CONFIG.IMG_SIZE,
                half=CONFIG.USE_FP16,
                verbose=False
            )

            current_detections = []
            for r in results:
                for box in r.boxes.xyxy:
                    current_detections.append(tuple(map(int, box)))

            # ── Tracking ─────────────────────────────────────────
            active_pids = self.tracker.update(current_detections, self.frame_count)

            # ── OCR + Drawing ────────────────────────────────────
            plates_processed = 0

            for pid in active_pids:
                if plates_processed >= CONFIG.MAX_PLATES_PER_FRAME:
                    break

                track = self.tracker.tracks[pid]
                x1, y1, x2, y2 = track["bbox"]
                crop = frame[y1:y2, x1:x2]

                # Skip tiny detections
                plate_area = (x2 - x1) * (y2 - y1)
                if plate_area < CONFIG.MIN_PLATE_AREA:
                    continue

                should_ocr = (self.frame_count - track["last_ocr"]) > CONFIG.OCR_COOLDOWN_FRAMES
                plate_text = track["text"]

                if should_ocr and crop.size > 0:
                    processed = preprocess_plate(crop)
                    if processed is not None:
                        try:
                            ocr_results = self.ocr_reader.readtext(
                                processed,
                                detail=1,
                                paragraph=False,
                                allowlist=CONFIG.OCR_ALLOWLIST,
                                contrast_ths=0.1,
                                adjust_contrast=0.5,
                                text_threshold=0.7,
                                low_text=0.4
                            )

                            candidates = []
                            for res in ocr_results:
                                _, text, conf = res
                                if conf >= CONFIG.OCR_CONF_THRESHOLD:
                                    cleaned = clean_plate_text(text)
                                    if cleaned:
                                        candidates.append((cleaned, conf))

                            if candidates:
                                # Best = highest confidence, tiebreaker = longest
                                best = max(candidates, key=lambda x: (x[1], len(x[0])))
                                track["text"] = best[0]
                                track["last_ocr"] = self.frame_count
                                plate_text = best[0]

                        except Exception as e:
                            print(f"OCR failed on plate #{pid}: {e}")

                # ── Draw ─────────────────────────────────────────
                color = (0, 255, 0) if plate_text else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"#{pid}"
                if plate_text:
                    label += f"  {plate_text}"
                cv2.putText(frame, label, (x1, y1 - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

                plates_processed += 1

            # ── FPS display ──────────────────────────────────────
            elapsed = time.time() - start_time
            instant_fps = 1 / elapsed if elapsed > 0 else 0
            self.fps_history.append(instant_fps)
            if len(self.fps_history) > CONFIG.FPS_SMOOTHING:
                self.fps_history.pop(0)
            avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0

            cv2.putText(frame, f"FPS: {int(avg_fps)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 2)

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