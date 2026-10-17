"""
Face and Multi-ROI Tracker using Relative Facial Geometry.
Extracts Forehead, Left Cheek, and Right Cheek ROIs dynamically scaled
to detected face bounding box fractions (no hardcoded pixel sizes).
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, Any, List

class FacialROITracker:
    """
    Robust relative-fraction facial ROI tracker with temporal Kalman/EMA smoothing.
    Gracefully scales from 1080p down to 240p (QVGA) without resolution collapse.
    """
    def __init__(self, smoothing_alpha: float = 0.7):
        self.smoothing_alpha = smoothing_alpha
        self.prev_face_box: Optional[Tuple[int, int, int, int]] = None
        
        # Load OpenCV default frontal face Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            # Fallback path if cv2.data is not configured
            self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def detect_face(self, frame_rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects primary face bounding box [x, y, w, h].
        """
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        # Multi-scale detection
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=4, 
            minSize=(24, 24)
        )
        if len(faces) == 0:
            return None
        
        # Pick the largest face in the frame (target subject)
        largest_face = max(faces, key=lambda b: b[2] * b[3])
        return tuple(int(v) for v in largest_face)

    def track_and_extract(self, frame_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Extracts Forehead, Left Cheek, and Right Cheek ROIs and bounding boxes.
        Returns extracted pixel arrays, normalized bounding boxes, and confidence.
        """
        h_frame, w_frame = frame_rgb.shape[:2]
        detected_box = self.detect_face(frame_rgb)
        
        if detected_box is not None:
            if self.prev_face_box is not None:
                # Exponential moving average smoothing on bounding box
                fx = int(self.smoothing_alpha * detected_box[0] + (1 - self.smoothing_alpha) * self.prev_face_box[0])
                fy = int(self.smoothing_alpha * detected_box[1] + (1 - self.smoothing_alpha) * self.prev_face_box[1])
                fw = int(self.smoothing_alpha * detected_box[2] + (1 - self.smoothing_alpha) * self.prev_face_box[2])
                fh = int(self.smoothing_alpha * detected_box[3] + (1 - self.smoothing_alpha) * self.prev_face_box[3])
                face_box = (fx, fy, fw, fh)
            else:
                face_box = detected_box
            self.prev_face_box = face_box
            confidence = 0.90
        elif self.prev_face_box is not None:
            # Maintain previous tracking state if single frame dropped
            face_box = self.prev_face_box
            confidence = 0.50
        else:
            # Heuristic central face box fallback if face never detected
            fw = int(w_frame * 0.45)
            fh = int(h_frame * 0.55)
            fx = int((w_frame - fw) / 2)
            fy = int(h_frame * 0.20)
            face_box = (fx, fy, fw, fh)
            confidence = 0.25

        fx, fy, fw, fh = face_box

        # Compute relative-fraction sub-ROIs (invariant to resolution C4)
        # 1. Forehead: Top-central region
        fh_w = int(fw * 0.45)
        fh_h = int(fh * 0.18)
        fh_x = fx + int(fw * 0.275)
        fy_offset = max(0, fy + int(fh * 0.10))
        fh_y = fy_offset

        # 2. Left Cheek (anatomical left / viewer right):
        lc_w = int(fw * 0.22)
        lc_h = int(fh * 0.22)
        lc_x = fx + int(fw * 0.15)
        lc_y = fy + int(fh * 0.52)

        # 3. Right Cheek (anatomical right / viewer left):
        rc_w = int(fw * 0.22)
        rc_h = int(fh * 0.22)
        rc_x = fx + int(fw * 0.63)
        rc_y = fy + int(fh * 0.52)

        def safe_crop(x: int, y: int, w: int, h: int) -> Tuple[np.ndarray, List[int]]:
            x1 = max(0, min(w_frame - 1, x))
            y1 = max(0, min(h_frame - 1, y))
            x2 = max(x1 + 1, min(w_frame, x1 + w))
            y2 = max(y1 + 1, min(h_frame, y1 + h))
            crop = frame_rgb[y1:y2, x1:x2]
            return crop, [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

        fh_crop, fh_bbox = safe_crop(fh_x, fh_y, fh_w, fh_h)
        lc_crop, lc_bbox = safe_crop(lc_x, lc_y, lc_w, lc_h)
        rc_crop, rc_bbox = safe_crop(rc_x, rc_y, rc_w, rc_h)

        # Compute spatial mean RGB per ROI
        fh_mean = np.mean(fh_crop, axis=(0, 1)) if fh_crop.size > 0 else np.zeros(3)
        lc_mean = np.mean(lc_crop, axis=(0, 1)) if lc_crop.size > 0 else np.zeros(3)
        rc_mean = np.mean(rc_crop, axis=(0, 1)) if rc_crop.size > 0 else np.zeros(3)

        return {
            "face_box": list(face_box),
            "rois": {
                "forehead": fh_crop,
                "left_cheek": lc_crop,
                "right_cheek": rc_crop
            },
            "bboxes": {
                "forehead": fh_bbox,
                "left_cheek": lc_bbox,
                "right_cheek": rc_bbox
            },
            "mean_rgb": {
                "forehead": fh_mean,
                "left_cheek": lc_mean,
                "right_cheek": rc_mean
            },
            "confidence": confidence
        }
