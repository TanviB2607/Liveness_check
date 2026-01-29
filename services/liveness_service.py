# import cv2
# import mediapipe as mp
# import numpy as np


# class ActiveLivenessTracker:
#     """
#     Streaming liveness (interactive):
#       WAIT_LEFT  -> user looks LEFT (sustained)
#       WAIT_RIGHT -> then looks RIGHT (sustained)
#       WAIT_BLINK -> then blinks (>=1)

#     update(frame_bgr, i) returns:
#       ("IN_PROGRESS", meta) or ("PASSED", meta)
#     finalize() returns:
#       ("FAILED", meta)
#     """

#     def __init__(
#         self,
#         face_mesh,
#         mirror: bool = True,
#         baseline_frames: int = 8,
#         delta: float = 0.10,
#         sustain_frames: int = 4,
#         min_gap_frames: int = 6,
#         min_valid_frames: int = 12,
#         debug: bool = False,
#     ):
#         self.mirror = mirror
#         self.face_mesh = face_mesh
#         self.BASELINE_FRAMES = baseline_frames
#         self.DELTA = delta
#         self.SUSTAIN_FRAMES = sustain_frames
#         self.MIN_GAP_FRAMES = min_gap_frames
#         self.MIN_VALID_FRAMES = min_valid_frames
#         self.debug = debug

#         self.state = "WAIT_LEFT"
#         self.sustain = 0
#         self.last_event_i = -10**9

#         self.offsets = []  # (i, norm_offset)
#         self.baseline = None
#         self.left_thr = None
#         self.right_thr = None

#         self.left_at = None
#         self.right_at = None

#         self.blink_count = 0
#         self.eye_closed = False
#         self.ear_after_right = []

#     @staticmethod
#     def _dist(a, b):
#         return float(np.hypot(a.x - b.x, a.y - b.y))

#     def _eye_ear_ratio(self, lm, left=True):
#         # mediapipe eye landmarks
#         if left:
#             outer, inner, upper, lower = 33, 133, 159, 145
#         else:
#             outer, inner, upper, lower = 263, 362, 386, 374

#         horiz = self._dist(lm[outer], lm[inner])
#         vert = self._dist(lm[upper], lm[lower])
#         if horiz <= 1e-6:
#             return None
#         return vert / horiz

#     def _ensure_baseline(self):
#         if self.baseline is not None:
#             return
#         if len(self.offsets) >= self.BASELINE_FRAMES:
#             early = [v for _, v in self.offsets[: self.BASELINE_FRAMES]]
#             self.baseline = float(np.median(early)) if early else 0.0
#             self.left_thr = self.baseline - self.DELTA
#             self.right_thr = self.baseline + self.DELTA
#             if self.debug:
#                 print("BASELINE", self.baseline, self.left_thr, self.right_thr)

#     def update(self, frame_bgr, i: int):
#         if self.mirror:
#             frame_bgr = cv2.flip(frame_bgr, 1)
#         rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#         results = self.face_mesh.process(rgb)
#         if not results.multi_face_landmarks:
#             return "IN_PROGRESS", {"state": self.state, "reason": "no_face"}

#         lm = results.multi_face_landmarks[0].landmark

#         # head turn: nose offset relative to eye midpoint normalized by eye distance
#         left_eye = lm[33]
#         right_eye = lm[263]
#         nose = lm[1]

#         eye_mid_x = (left_eye.x + right_eye.x) / 2.0
#         eye_dist = abs(right_eye.x - left_eye.x)
#         if eye_dist > 1e-6:
#             norm_offset = float((nose.x - eye_mid_x) / eye_dist)
#             self.offsets.append((i, norm_offset))

#         # ear for blink
#         ear_l = self._eye_ear_ratio(lm, left=True)
#         ear_r = self._eye_ear_ratio(lm, left=False)
#         ear = None
#         if ear_l is not None and ear_r is not None:
#             ear = float((ear_l + ear_r) / 2.0)
#             if not (0.02 <= ear <= 0.75):
#                 ear = None

#         self._ensure_baseline()

#         if self.baseline is None or len(self.offsets) < self.MIN_VALID_FRAMES:
#             return "IN_PROGRESS", {"state": self.state, "reason": "collecting"}

#         off = self.offsets[-1][1]

#         if self.state == "WAIT_LEFT":
#             self.sustain = self.sustain + 1 if off < self.left_thr else 0
#             if self.sustain >= self.SUSTAIN_FRAMES and (i - self.last_event_i >= self.MIN_GAP_FRAMES):
#                 self.left_at = i
#                 self.last_event_i = i
#                 self.state = "WAIT_RIGHT"
#                 self.sustain = 0

#         elif self.state == "WAIT_RIGHT":
#             self.sustain = self.sustain + 1 if off > self.right_thr else 0
#             if self.sustain >= self.SUSTAIN_FRAMES and (i - self.last_event_i >= self.MIN_GAP_FRAMES):
#                 self.right_at = i
#                 self.last_event_i = i
#                 self.state = "WAIT_BLINK"
#                 self.sustain = 0
#                 self.blink_count = 0
#                 self.eye_closed = False
#                 self.ear_after_right = []

#         elif self.state == "WAIT_BLINK":
#             if ear is not None and self.right_at is not None and i >= self.right_at:
#                 self.ear_after_right.append(ear)
#                 if len(self.ear_after_right) > 60:
#                     self.ear_after_right = self.ear_after_right[-60:]

#                 if len(self.ear_after_right) >= 10:
#                     baseline_open = float(np.percentile(self.ear_after_right, 70))
#                     blink_thr = baseline_open * 0.75

#                     if ear < blink_thr and not self.eye_closed:
#                         self.eye_closed = True
#                     elif ear >= blink_thr and self.eye_closed:
#                         self.blink_count += 1
#                         self.eye_closed = False

#                     if self.blink_count >= 1:
#                         return "PASSED", {
#                             "state": self.state,
#                             "left_at": self.left_at,
#                             "right_at": self.right_at,
#                             "blink_count": self.blink_count,
#                         }

#         return "IN_PROGRESS", {
#             "state": self.state,
#             "left_at": self.left_at,
#             "right_at": self.right_at,
#             "blink_count": self.blink_count,
#         }

#     def finalize(self):
#         return "FAILED", {
#             "state": self.state,
#             "left_at": self.left_at,
#             "right_at": self.right_at,
#             "blink_count": self.blink_count,
#         }


# class LivenessService:
#     """Holds MediaPipe FaceMesh so it loads once."""
#     def __init__(self):
#         self.mp_face_mesh = mp.solutions.face_mesh
#         self.face_mesh = self.mp_face_mesh.FaceMesh(
#             static_image_mode=False,
#             max_num_faces=1,
#             refine_landmarks=True,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         )


#OG CODE
# import cv2
# import mediapipe as mp
# import numpy as np
# import random
# import time


# class ActiveLivenessTracker:
#     ACTIONS = ["LEFT", "RIGHT", "BLINK", "SMILE", "NOD"]

#     def __init__(
#         self,
#         face_mesh,
#         mirror=True,
#         baseline_frames=8,
#         delta=0.10,
#         sustain_frames=4,
#         min_gap_frames=6,
#         min_valid_frames=12,
#         debug=False,
#     ):
#         self.face_mesh = face_mesh
#         self.mirror = mirror

#         self.BASELINE_FRAMES = baseline_frames
#         self.DELTA = delta
#         self.SUSTAIN_FRAMES = sustain_frames
#         self.MIN_GAP_FRAMES = min_gap_frames
#         self.MIN_VALID_FRAMES = min_valid_frames
#         self.debug = debug

#         # RANDOM ACTION ORDER
#         self.action_queue = random.sample(self.ACTIONS, len(self.ACTIONS))
#         self.current_action = self.action_queue.pop(0)

#         self.sustain = 0
#         self.last_event_i = -99999

#         self.offsets_x = []
#         self.offsets_y = []

#         self.baseline_x = None
#         self.baseline_y = None

#         self.left_thr = None
#         self.right_thr = None
#         self.up_thr = None
#         self.down_thr = None

#         self.blink_count = 0
#         self.eye_closed = False
#         self.ear_buffer = []

#         self.smile_count = 0
#         self.mouth_ratio_buffer = []

#         # IDLE CHECK
#         self.ACTION_TIMEOUT = 10  # seconds
#         self.action_start_time = time.time()

#     # --------------------------------------------------

#     @staticmethod
#     def _dist(a, b):
#         return float(np.hypot(a.x - b.x, a.y - b.y))

#     def _eye_ear_ratio(self, lm, left=True):
#         if left:
#             outer, inner, upper, lower = 33, 133, 159, 145
#         else:
#             outer, inner, upper, lower = 263, 362, 386, 374

#         horiz = self._dist(lm[outer], lm[inner])
#         vert = self._dist(lm[upper], lm[lower])

#         if horiz <= 1e-6:
#             return None

#         return vert / horiz

#     def _mouth_ratio(self, lm):
#         left = lm[61]
#         right = lm[291]
#         top = lm[13]
#         bottom = lm[14]

#         horiz = self._dist(left, right)
#         vert = self._dist(top, bottom)

#         if horiz <= 1e-6:
#             return None

#         return vert / horiz

#     # --------------------------------------------------

#     def _ensure_baseline(self):
#         if self.baseline_x is None and len(self.offsets_x) >= self.BASELINE_FRAMES:
#             early = [v for _, v in self.offsets_x[: self.BASELINE_FRAMES]]
#             self.baseline_x = float(np.median(early))
#             self.left_thr = self.baseline_x - self.DELTA
#             self.right_thr = self.baseline_x + self.DELTA

#         if self.baseline_y is None and len(self.offsets_y) >= self.BASELINE_FRAMES:
#             early = [v for _, v in self.offsets_y[: self.BASELINE_FRAMES]]
#             self.baseline_y = float(np.median(early))
#             self.up_thr = self.baseline_y - self.DELTA
#             self.down_thr = self.baseline_y + self.DELTA

#     # --------------------------------------------------

#     def _next_action(self):
#         if not self.action_queue:
#             self.current_action = None
#             return

#         self.current_action = self.action_queue.pop(0)
#         self.action_start_time = time.time()

#         self.sustain = 0
#         self.blink_count = 0
#         self.eye_closed = False
#         self.ear_buffer.clear()

#         self.smile_count = 0
#         self.mouth_ratio_buffer.clear()

#     # --------------------------------------------------

#     def update(self, frame, i):
#         # FAIL if action takes too long
#         if time.time() - self.action_start_time > self.ACTION_TIMEOUT:
#             return "FAILED", {
#                 "reason": "ACTION_TIMEOUT",
#                 "pending_action": self.current_action,
#             }

#         if self.mirror:
#             frame = cv2.flip(frame, 1)

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         res = self.face_mesh.process(rgb)

#         if not res.multi_face_landmarks:
#             return "IN_PROGRESS", {
#                 "action": self.current_action,
#                 "reason": "no_face",
#             }

#         lm = res.multi_face_landmarks[0].landmark

#         # ------------------------------------
#         # HEAD MOVEMENT
#         left_eye = lm[33]
#         right_eye = lm[263]
#         nose = lm[1]

#         eye_mid_x = (left_eye.x + right_eye.x) / 2
#         eye_mid_y = (left_eye.y + right_eye.y) / 2
#         eye_dist = abs(right_eye.x - left_eye.x)

#         if eye_dist > 1e-6:
#             off_x = (nose.x - eye_mid_x) / eye_dist
#             off_y = (nose.y - eye_mid_y) / eye_dist

#             self.offsets_x.append((i, off_x))
#             self.offsets_y.append((i, off_y))

#         # ------------------------------------
#         # BLINK
#         ear_l = self._eye_ear_ratio(lm, True)
#         ear_r = self._eye_ear_ratio(lm, False)
#         ear = (ear_l + ear_r) / 2 if ear_l and ear_r else None

#         # ------------------------------------
#         # SMILE
#         mouth_ratio = self._mouth_ratio(lm)

#         self._ensure_baseline()
#         if self.baseline_x is None:
#             return "IN_PROGRESS", {
#                 "action": self.current_action,
#                 "reason": "calibrating",
#             }

#         off_x = self.offsets_x[-1][1]
#         off_y = self.offsets_y[-1][1]

#         # --------------------------------------------------
#         if self.current_action == "LEFT":
#             self.sustain = self.sustain + 1 if off_x < self.left_thr else 0
#             if self.sustain >= self.SUSTAIN_FRAMES:
#                 self._next_action()
#                 return "IN_PROGRESS", {"action_done": True}

#         elif self.current_action == "RIGHT":
#             self.sustain = self.sustain + 1 if off_x > self.right_thr else 0
#             if self.sustain >= self.SUSTAIN_FRAMES:
#                 self._next_action()
#                 return "IN_PROGRESS", {"action_done": True}

#         elif self.current_action == "NOD":
#             nod_threshold = self.down_thr * 0.8
#             self.sustain = self.sustain + 1 if off_y > nod_threshold else 0
#             if self.sustain >= self.SUSTAIN_FRAMES:
#                 self._next_action()
#                 return "IN_PROGRESS", {"action_done": True}

#         elif self.current_action == "BLINK" and ear:
#             self.ear_buffer.append(ear)
#             if len(self.ear_buffer) > 30:
#                 self.ear_buffer = self.ear_buffer[-30:]

#             if len(self.ear_buffer) >= 10:
#                 base = np.percentile(self.ear_buffer, 70)
#                 thr = base * 0.75

#                 if ear < thr and not self.eye_closed:
#                     self.eye_closed = True
#                 elif ear >= thr and self.eye_closed:
#                     self.blink_count += 1
#                     self.eye_closed = False

#                 if self.blink_count >= 1:
#                     self._next_action()
#                     return "IN_PROGRESS", {"action_done": True}

#         elif self.current_action == "SMILE" and mouth_ratio:
#             self.mouth_ratio_buffer.append(mouth_ratio)
#             if len(self.mouth_ratio_buffer) > 30:
#                 self.mouth_ratio_buffer = self.mouth_ratio_buffer[-30:]

#             if len(self.mouth_ratio_buffer) >= 10:
#                 baseline = np.mean(self.mouth_ratio_buffer[:5])
#                 if mouth_ratio > baseline * 1.15:
#                     self.smile_count += 1

#                 if self.smile_count >= 3:
#                     self._next_action()
#                     return "IN_PROGRESS", {"action_done": True}

#         # --------------------------------------------------
#         if self.current_action is None:
#             return "PASSED", {"status": "LIVENESS_PASSED"}

#         return "IN_PROGRESS", {
#             "action": self.current_action,
#             "queue_left": self.action_queue,
#         }

#     # --------------------------------------------------

#     def finalize(self):
#         return "FAILED", {
#             "pending_action": self.current_action,
#             "queue_left": self.action_queue,
#         }


# # =========================================================

# class LivenessService:
#     """Loads MediaPipe once"""

#     def __init__(self):
#         self.mp_face_mesh = mp.solutions.face_mesh
#         self.face_mesh = self.mp_face_mesh.FaceMesh(
#             static_image_mode=False,
#             max_num_faces=1,
#             refine_landmarks=True,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5,
#         )

#     def create_tracker(self):
#         return ActiveLivenessTracker(self.face_mesh)





import cv2
import mediapipe as mp
import numpy as np
import random
import time


class ActiveLivenessTracker:
    ACTIONS = ["LEFT", "RIGHT", "BLINK", "SMILE", "NOD"]

    def __init__(
        self,
        face_mesh,
        mirror=True,
        baseline_frames=8,
        delta=0.10,
        sustain_frames=4,
        min_gap_frames=6,
        min_valid_frames=12,
        debug=False,
    ):
        self.face_mesh = face_mesh
        self.mirror = mirror

        self.BASELINE_FRAMES = baseline_frames
        self.DELTA = delta
        self.SUSTAIN_FRAMES = sustain_frames
        self.MIN_GAP_FRAMES = min_gap_frames
        self.MIN_VALID_FRAMES = min_valid_frames
        self.debug = debug

        # RANDOM ACTION ORDER
        self.action_queue = random.sample(self.ACTIONS, len(self.ACTIONS))
        self.current_action = self.action_queue.pop(0)

        self.sustain = 0
        self.last_event_i = -99999

        self.offsets_x = []
        self.offsets_y = []

        self.baseline_x = None
        self.baseline_y = None

        self.left_thr = None
        self.right_thr = None
        self.up_thr = None
        self.down_thr = None

        self.blink_count = 0
        self.eye_closed = False
        self.ear_buffer = []

        self.smile_count = 0
        self.mouth_ratio_buffer = []

        # IDLE CHECK
        self.ACTION_TIMEOUT = 10  # seconds
        self.action_start_time = time.time()

        # ---------------- SPOOF DETECTION ----------------
        self.spoof_score = 0
        self.SPOOF_THRESHOLD = 5

        # Depth history
        self.depth_buffer = []

        # Optical flow
        self.prev_gray = None
        self.flow_points = None

    # --------------------------------------------------

    @staticmethod
    def _dist(a, b):
        return float(np.hypot(a.x - b.x, a.y - b.y))

    def _eye_ear_ratio(self, lm, left=True):
        if left:
            outer, inner, upper, lower = 33, 133, 159, 145
        else:
            outer, inner, upper, lower = 263, 362, 386, 374

        horiz = self._dist(lm[outer], lm[inner])
        vert = self._dist(lm[upper], lm[lower])

        if horiz <= 1e-6:
            return None

        return vert / horiz

    def _mouth_ratio(self, lm):
        left = lm[61]
        right = lm[291]
        top = lm[13]
        bottom = lm[14]

        horiz = self._dist(left, right)
        vert = self._dist(top, bottom)

        if horiz <= 1e-6:
            return None

        return vert / horiz

    # --------------------------------------------------

    def _ensure_baseline(self):
        if self.baseline_x is None and len(self.offsets_x) >= self.BASELINE_FRAMES:
            early = [v for _, v in self.offsets_x[: self.BASELINE_FRAMES]]
            self.baseline_x = float(np.median(early))
            self.left_thr = self.baseline_x - self.DELTA
            self.right_thr = self.baseline_x + self.DELTA

        if self.baseline_y is None and len(self.offsets_y) >= self.BASELINE_FRAMES:
            early = [v for _, v in self.offsets_y[: self.BASELINE_FRAMES]]
            self.baseline_y = float(np.median(early))
            self.up_thr = self.baseline_y - self.DELTA
            self.down_thr = self.baseline_y + self.DELTA

    # --------------------------------------------------

    def _next_action(self):
        if not self.action_queue:
            self.current_action = None
            return

        self.current_action = self.action_queue.pop(0)
        self.action_start_time = time.time()

        self.sustain = 0
        self.blink_count = 0
        self.eye_closed = False
        self.ear_buffer.clear()

        self.smile_count = 0
        self.mouth_ratio_buffer.clear()

    # --------------------------------------------------

    def update(self, frame, i):
        # FAIL if action takes too long
        if time.time() - self.action_start_time > self.ACTION_TIMEOUT:
            return "FAILED", {
                "reason": "ACTION_TIMEOUT",
                "pending_action": self.current_action,
            }

        if self.mirror:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            return "IN_PROGRESS", {
                "action": self.current_action,
                "reason": "no_face",
            }

        lm = res.multi_face_landmarks[0].landmark

        # ------------------------------------
        # HEAD MOVEMENT
        left_eye = lm[33]
        right_eye = lm[263]
        nose = lm[1]

        eye_mid_x = (left_eye.x + right_eye.x) / 2
        eye_mid_y = (left_eye.y + right_eye.y) / 2
        eye_dist = abs(right_eye.x - left_eye.x)

        if eye_dist > 1e-6:
            off_x = (nose.x - eye_mid_x) / eye_dist
            off_y = (nose.y - eye_mid_y) / eye_dist

            self.offsets_x.append((i, off_x))
            self.offsets_y.append((i, off_y))

        # ------------------------------------
        # BLINK
        ear_l = self._eye_ear_ratio(lm, True)
        ear_r = self._eye_ear_ratio(lm, False)
        ear = (ear_l + ear_r) / 2 if ear_l and ear_r else None

        # ------------------------------------
        # SMILE
        mouth_ratio = self._mouth_ratio(lm)

        # ---------------- SPOOF CHECKS ----------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self._check_depth_flatness(lm)
        self._check_optical_flow(gray)

        if self.spoof_score >= self.SPOOF_THRESHOLD:
            return "FAILED", {
            "reason": "SPOOF_DETECTED",
            "score": self.spoof_score
        }


        self._ensure_baseline()
        if self.baseline_x is None:
            return "IN_PROGRESS", {
                "action": self.current_action,
                "reason": "calibrating",
            }

        off_x = self.offsets_x[-1][1]
        off_y = self.offsets_y[-1][1]

        # --------------------------------------------------
        if self.current_action == "LEFT":
            self.sustain = self.sustain + 1 if off_x < self.left_thr else 0
            if self.sustain >= self.SUSTAIN_FRAMES:
                self._next_action()
                return "IN_PROGRESS", {"action_done": True}

        elif self.current_action == "RIGHT":
            self.sustain = self.sustain + 1 if off_x > self.right_thr else 0
            if self.sustain >= self.SUSTAIN_FRAMES:
                self._next_action()
                return "IN_PROGRESS", {"action_done": True}

        elif self.current_action == "NOD":
            nod_threshold = self.down_thr * 0.75
            self.sustain = self.sustain + 1 if off_y > nod_threshold else 0
            if self.sustain >= self.SUSTAIN_FRAMES:
                self._next_action()
                return "IN_PROGRESS", {"action_done": True}

        elif self.current_action == "BLINK" and ear:
            self.ear_buffer.append(ear)
            if len(self.ear_buffer) > 30:
                self.ear_buffer = self.ear_buffer[-30:]

            if len(self.ear_buffer) >= 10:
                base = np.percentile(self.ear_buffer, 70)
                thr = base * 0.75

                if ear < thr and not self.eye_closed:
                    self.eye_closed = True
                elif ear >= thr and self.eye_closed:
                    self.blink_count += 1
                    self.eye_closed = False

                if self.blink_count >= 1:
                    self._next_action()
                    return "IN_PROGRESS", {"action_done": True}

        elif self.current_action == "SMILE" and mouth_ratio:
            self.mouth_ratio_buffer.append(mouth_ratio)
            if len(self.mouth_ratio_buffer) > 30:
                self.mouth_ratio_buffer = self.mouth_ratio_buffer[-30:]

            if len(self.mouth_ratio_buffer) >= 10:
                baseline = np.mean(self.mouth_ratio_buffer[:5])
                if mouth_ratio > baseline * 1.15:
                    self.smile_count += 1

                if self.smile_count >= 3:
                    self._next_action()
                    return "IN_PROGRESS", {"action_done": True}

        # --------------------------------------------------
        if self.current_action is None:
            return "PASSED", {"status": "LIVENESS_PASSED"}

        return "IN_PROGRESS", {
            "action": self.current_action,
            "queue_left": self.action_queue,
        }

    # --------------------------------------------------

    def finalize(self):
        return "FAILED", {
            "pending_action": self.current_action,
            "queue_left": self.action_queue,
        }
    
    def _check_depth_flatness(self, lm):
        nose_z = lm[1].z
        left_ear_z = lm[234].z
        right_ear_z = lm[454].z

        ear_mean_z = (left_ear_z + right_ear_z) / 2
        nose_depth_gain = abs(nose_z - ear_mean_z)

        self.depth_buffer.append(nose_depth_gain)
        if len(self.depth_buffer) > 15:
            self.depth_buffer = self.depth_buffer[-15:]

        # Flat screens have very low depth variance
        if len(self.depth_buffer) >= 10:
            mean_depth = np.mean(self.depth_buffer)
            if mean_depth < 0.015:  # 🔴 tuned for webcams
                self.spoof_score += 1

    def _check_optical_flow(self, frame_gray):
    
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            return

        # Track nose + cheek regions
        h, w = frame_gray.shape
        points = np.array([
            [int(w * 0.5), int(h * 0.5)],      # nose
            [int(w * 0.35), int(h * 0.55)],    # left cheek
            [int(w * 0.65), int(h * 0.55)],    # right cheek
        ], dtype=np.float32).reshape(-1, 1, 2)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            frame_gray,
            points,
            None,
            winSize=(15, 15),
            maxLevel=2
        )

        if next_pts is not None and status.sum() >= 2:
            motion = next_pts - points
            variance = np.var(motion)

            # Screens move as a single rigid plane
            if variance < 1e-4:  # 🔴 tuned threshold
                self.spoof_score += 1

        self.prev_gray = frame_gray


# =========================================================

class LivenessService:
    """Loads MediaPipe once"""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def create_tracker(self):
        return ActiveLivenessTracker(self.face_mesh)
