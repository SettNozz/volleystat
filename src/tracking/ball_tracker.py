import numpy as np

class BallTracker:
    def __init__(self):
        self.trajectory = []  # List of (frame_idx, x, y)

    def update(self, frame_idx, detection):
        """
        Update tracker with new detection.
        detection: (x, y) or (x, y, score)
        """
        if detection is not None:
            if len(detection) == 2:
                x, y = detection
            else:
                x, y = detection[:2]
            self.trajectory.append((frame_idx, x, y))

    def get_trajectory(self):
        return self.trajectory