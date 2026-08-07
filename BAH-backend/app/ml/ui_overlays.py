import cv2
import numpy as np
import io

def draw_tracking_overlay(rgb_bytes: bytes, eye_pos, history, is_interp, speed_label=""):
    """
    Draws the tracking UI overlay on a PNG frame.
    eye_pos: (x, y)
    history: list of (x, y) tuples of previous real frames
    is_interp: bool, whether this is a real or interpolated frame
    speed_label: str to annotate
    """
    np_arr = np.frombuffer(rgb_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # 1. Draw Trajectory (persistent white polyline)
    if len(history) > 1:
        pts = np.array(history, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=False, color=(255, 255, 255), thickness=1)
        
    # Connect last history to current eye
    if len(history) > 0 and eye_pos is not None:
        cv2.line(img, history[-1], eye_pos, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Draw Marker
    if eye_pos is not None:
        x, y = eye_pos
        if is_interp:
            # Blue Triangle
            size = 6
            pts = np.array([[x, y - size], [x - size, y + size], [x + size, y + size]], np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(255, 100, 50), thickness=2) # BGR
            cv2.fillPoly(img, [pts], color=(255, 0, 0))
        else:
            # Green Circle
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), 6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Motion Arrow (red) if history exists to compute direction
            # We'll just draw a generic label for speed
            if speed_label:
                cv2.putText(img, speed_label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    success, encoded = cv2.imencode(".png", img)
    if not success:
        return rgb_bytes
    return encoded.tobytes()
