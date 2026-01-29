import numpy as np

class SpoofDetector:
    def __init__(self):
        self.depth_history = []

    def predict(self, landmarks):

        nose = landmarks[1]
        left = landmarks[234]
        right = landmarks[454]

        depth = abs(nose.z - (left.z + right.z) / 2)
        self.depth_history.append(depth)

        if len(self.depth_history) > 12:
            self.depth_history = self.depth_history[-12:]

        if len(self.depth_history) >= 6:
            variance = np.std(self.depth_history)
            print("DEPTH:", round(depth, 6), "VAR:", round(variance, 6))

            # 🔥 tuned using your real logs
            if variance < 0.002:
                return True, variance

        return False, depth
