"""Real-time driver drowsiness detection with a simulated auto-braking response.

Detect -> decide -> act pipeline:
  detect : eye aspect ratio (EAR) + calibrated head-pitch deviation (MediaPipe Face Mesh + solvePnP)
  decide : either signal sustained for ALERT_SECONDS => drowsy
  act    : audible alert + simulated vehicle decelerates (auto-braking)

Keys:  m = toggle vehicle moving/stopped   r = recalibrate head pose   q = quit
"""
import math
import platform
import sys
import threading
import time

import cv2
import mediapipe as mp
import numpy as np

# ------------------------------ configuration ------------------------------
EAR_THRESHOLD = 0.25          # below this the eye counts as closed
PITCH_DEVIATION_DEG = 15      # head pitch deviation from calibrated baseline
ALERT_SECONDS = 1.0           # signal must persist this long (frame-rate independent)
CALIBRATION_FRAMES = 30

MAX_SPEED = 60.0              # km/h
BRAKE_RATE = 2.0              # km/h lost per frame while drowsy
RECOVER_RATE = 1.0            # km/h gained per frame when alert

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# generic 3D face model used for head pose
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),           # nose tip
    (0.0, -330.0, -65.0),      # chin
    (-225.0, 170.0, -135.0),   # left eye outer corner
    (225.0, 170.0, -135.0),    # right eye outer corner
    (-150.0, -150.0, -125.0),  # left mouth corner
    (150.0, -150.0, -125.0),   # right mouth corner
], dtype=np.float64)
POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]


# ------------------------------ detection helpers ------------------------------
def calculate_ear(landmarks, eye_indices, w, h):
    p1, p2, p3, p4, p5, p6 = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    vertical = math.dist(p2, p6) + math.dist(p3, p5)
    return vertical / (2.0 * math.dist(p1, p4))


def get_head_pitch(landmarks, w, h):
    """Head pitch in degrees, or None if pose estimation fails."""
    image_points = np.array([(landmarks[i].x * w, landmarks[i].y * h)
                             for i in POSE_LANDMARKS], dtype=np.float64)
    camera_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix,
                                  np.zeros((4, 1)))
    if not ok:
        return None
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    euler = cv2.decomposeProjectionMatrix(np.hstack((rotation_matrix, tvec)))[6]
    return float(euler[0][0])


def angle_diff(a, b):
    """Smallest signed difference a-b in degrees (handles the +/-180 wrap)."""
    return (a - b + 180.0) % 360.0 - 180.0


# ------------------------------ alert sound ------------------------------
def play_beep():
    try:
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(1000, 400)
        else:
            sys.stdout.write("\a")   # terminal bell fallback (macOS/Linux)
            sys.stdout.flush()
    except Exception:
        pass


# ------------------------------ vehicle simulation ------------------------------
def draw_vehicle_panel(frame, speed, braking):
    h, w, _ = frame.shape
    x, y = w - 260, h - 140
    cv2.rectangle(frame, (x, y), (w - 20, h - 20), (30, 30, 30), -1)
    cv2.rectangle(frame, (x, y), (w - 20, h - 20), (200, 200, 200), 2)

    cv2.rectangle(frame, (x + 30, y + 40), (x + 170, y + 75),
                  (0, 0, 255) if braking else (0, 200, 0), -1)
    cv2.circle(frame, (x + 55, y + 80), 10, (20, 20, 20), -1)
    cv2.circle(frame, (x + 145, y + 80), 10, (20, 20, 20), -1)
    cv2.circle(frame, (x + 170, y + 50), 6, (0, 0, 255) if braking else (60, 60, 60), -1)

    cv2.putText(frame, f"Speed: {speed:.0f} km/h", (x + 10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if braking:
        cv2.putText(frame, "AUTO-BRAKING", (x + 10, y + 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


def put(frame, text, pos, color=(0, 255, 0), scale=0.8, thick=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


# ------------------------------ main loop ------------------------------
def main():
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Could not open webcam.")

    calibration, baseline_pitch = [], None
    eyes_closed_since = head_off_since = None
    alert_active = False
    vehicle_moving = True
    speed = MAX_SPEED

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        now = time.time()
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        drowsy, reason = False, ""

        if results.multi_face_landmarks and vehicle_moving:
            lm = results.multi_face_landmarks[0].landmark
            ear = (calculate_ear(lm, LEFT_EYE, w, h) + calculate_ear(lm, RIGHT_EYE, w, h)) / 2
            pitch = get_head_pitch(lm, w, h)

            if baseline_pitch is None:
                if pitch is not None:
                    calibration.append(pitch)
                put(frame, f"Calibrating... sit normally ({len(calibration)}/{CALIBRATION_FRAMES})",
                    (30, 50), (0, 255, 255), 0.8)
                if len(calibration) >= CALIBRATION_FRAMES:
                    baseline_pitch = float(np.median(calibration))
                    print(f"Calibration done. Baseline pitch: {baseline_pitch:.1f} deg")
            else:
                eyes_closed_since = (eyes_closed_since or now) if ear < EAR_THRESHOLD else None
                deviation = abs(angle_diff(pitch, baseline_pitch)) if pitch is not None else 0.0
                head_off_since = (head_off_since or now) if deviation > PITCH_DEVIATION_DEG else None

                if eyes_closed_since and now - eyes_closed_since >= ALERT_SECONDS:
                    drowsy, reason = True, "EYES CLOSED"
                elif head_off_since and now - head_off_since >= ALERT_SECONDS:
                    drowsy, reason = True, "HEAD NOD"

                put(frame, f"EAR: {ear:.3f}", (30, 50), (0, 255, 0), 1)
                put(frame, f"Pitch dev: {deviation:.1f} deg", (30, 150), (255, 255, 0))
        else:
            # no face, or vehicle stopped: nothing accumulates
            eyes_closed_since = head_off_since = None
            if not vehicle_moving:
                put(frame, "Monitoring paused (vehicle stopped)", (30, 50), (200, 200, 0))
            else:
                put(frame, "No face detected", (30, 50), (0, 165, 255))

        if drowsy:
            put(frame, f"DROWSINESS ALERT! ({reason})", (30, 100), (0, 0, 255), 0.9, 3)
            if not alert_active:
                threading.Thread(target=play_beep, daemon=True).start()
            alert_active = True
        else:
            alert_active = False

        if vehicle_moving:
            speed = max(0.0, speed - BRAKE_RATE) if drowsy else min(MAX_SPEED, speed + RECOVER_RATE)
        else:
            speed = 0.0
        draw_vehicle_panel(frame, speed, braking=drowsy)

        put(frame, "m: moving/stopped   r: recalibrate   q: quit", (30, h - 20),
            (180, 180, 180), 0.5, 1)
        cv2.imshow("Drowsiness Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            vehicle_moving = not vehicle_moving
            if vehicle_moving:            # resuming: recalibrate for the driver's current posture
                calibration, baseline_pitch = [], None
        if key == ord("r"):
            calibration, baseline_pitch = [], None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()