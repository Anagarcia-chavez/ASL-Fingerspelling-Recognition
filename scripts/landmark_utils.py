"""
Shared helper for turning MediaPipe hand landmarks into a normalized
feature vector. This MUST be used identically during dataset extraction,
custom-gesture collection, and real-time inference, or the classifier
will see inconsistent features.
"""

import numpy as np

# MediaPipe hand landmark indices:
#   0  = wrist
#   9  = middle finger MCP (base knuckle of middle finger)
WRIST = 0
MIDDLE_MCP = 9


def normalize_landmarks(landmarks):
    """
    Args:
        landmarks: iterable of 21 (x, y, z) tuples/lists from MediaPipe
                   (values are already in MediaPipe's normalized 0-1 image
                   coordinates).

    Returns:
        numpy array of shape (63,) - flattened, translation- and
        scale-normalized landmark coordinates.

    Why this normalization:
      - Subtracting the wrist position makes the features independent of
        where the hand is in the camera frame.
      - Dividing by the wrist-to-middle-knuckle distance makes the features
        independent of hand size / distance from the camera.
      - We do NOT normalize rotation, because hand orientation is part of
        what distinguishes many ASL letters (e.g. 'M' vs 'N' vs 'T').
    """
    coords = np.array(landmarks, dtype=np.float32)  # shape (21, 3)

    wrist = coords[WRIST].copy()
    coords -= wrist

    scale = np.linalg.norm(coords[MIDDLE_MCP])
    if scale > 1e-6:
        coords /= scale

    return coords.flatten()  # 63 values: x0,y0,z0, x1,y1,z1, ...
