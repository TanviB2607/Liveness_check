import cv2
import numpy as np

class SpoofDetector:
    def __init__(self):
        self.prev_gray = None
        self.static_count = 0

    def predict(self, frame, static_threshold=15):
        """
        Returns:
            is_spoof (bool)
            confidence_real (float)
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -------- MOTION CHECK (anti-photo / anti-replay)
        if self.prev_gray is None:
            self.prev_gray = gray
            return False, 0.99

        diff = cv2.absdiff(self.prev_gray, gray)
        motion_score = np.mean(diff)

        self.prev_gray = gray

        if motion_score < static_threshold:
            self.static_count += 1
        else:
            self.static_count = 0

        # Too many static frames → spoof
        if self.static_count > 20:
            return True, 0.1

        return False, 0.95
