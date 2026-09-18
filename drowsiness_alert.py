import cv2
import mediapipe as mp
import math
import numpy as np
import winsound
import threading

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20
PITCH_DEVIATION_THRESHOLD = 15
NOD_CONSEC_FRAMES = 20
CALIBRATION_FRAMES = 30

closed_frame_count = 0
nod_frame_count = 0
alert_active = False

calibration_values = []
baseline_pitch = None

# simulated vehicle state
speed = 60.0
MAX_SPEED = 60.0
BRAKE_RATE = 2.0
RECOVER_RATE = 1.0

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)

LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def calculate_ear(landmarks, eye_indices, frame_w, frame_h):
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        points.append((lm.x * frame_w, lm.y * frame_h))
    p1, p2, p3, p4, p5, p6 = points
    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def get_head_pitch(landmarks, frame_w, frame_h):
    image_points = np.array([
        (landmarks[idx].x * frame_w, landmarks[idx].y * frame_h)
        for idx in LANDMARK_IDS
    ], dtype=np.float64)

    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs
    )
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    return euler_angles[0][0]


def play_beep():
    winsound.Beep(1000, 400)


def draw_vehicle_panel(frame, speed, braking):
    # panel background, bottom-right corner
    h, w, _ = frame.shape
    panel_x, panel_y = w - 260, h - 140
    cv2.rectangle(frame, (panel_x, panel_y), (w - 20, h - 20), (30, 30, 30), -1)
    cv2.rectangle(frame, (panel_x, panel_y), (w - 20, h - 20), (200, 200, 200), 2)

    # simple car body
    car_color = (0, 0, 255) if braking else (0, 200, 0)
    cv2.rectangle(frame, (panel_x + 30, panel_y + 40), (panel_x + 170, panel_y + 75), car_color, -1)
    cv2.circle(frame, (panel_x + 55, panel_y + 80), 10, (20, 20, 20), -1)
    cv2.circle(frame, (panel_x + 145, panel_y + 80), 10, (20, 20, 20), -1)

    # brake lights
    brake_color = (0, 0, 255) if braking else (60, 60, 60)
    cv2.circle(frame, (panel_x + 170, panel_y + 50), 6, brake_color, -1)

    cv2.putText(frame, f"Speed: {speed:.0f} km/h", (panel_x + 10, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if braking:
        cv2.putText(frame, "AUTO-BRAKING", (panel_x + 10, panel_y + 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    drowsy = False

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_ear = calculate_ear(landmarks, LEFT_EYE, frame_w, frame_h)
        right_ear = calculate_ear(landmarks, RIGHT_EYE, frame_w, frame_h)
        avg_ear = (left_ear + right_ear) / 2.0

        pitch = get_head_pitch(landmarks, frame_w, frame_h)

        if baseline_pitch is None:
            calibration_values.append(pitch)
            cv2.putText(frame, "Calibrating... sit normally", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            if len(calibration_values) >= CALIBRATION_FRAMES:
                baseline_pitch = sum(calibration_values) / len(calibration_values)
                print(f"Calibration done. Baseline pitch: {baseline_pitch:.1f}")
        else:
            pitch_deviation = abs(pitch - baseline_pitch)

            if avg_ear < EAR_THRESHOLD:
                closed_frame_count += 1
            else:
                closed_frame_count = 0

            if pitch_deviation > PITCH_DEVIATION_THRESHOLD:
                nod_frame_count += 1
            else:
                nod_frame_count = 0

            drowsy = closed_frame_count >= CONSEC_FRAMES or nod_frame_count >= NOD_CONSEC_FRAMES

            if not drowsy:
                alert_active = False

            if drowsy:
                reason = "EYES CLOSED" if closed_frame_count >= CONSEC_FRAMES else "HEAD NOD"
                cv2.putText(frame, f"DROWSINESS ALERT! ({reason})", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                if not alert_active:
                    threading.Thread(target=play_beep, daemon=True).start()
                    alert_active = True

            cv2.putText(frame, f"EAR: {avg_ear:.3f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Pitch dev: {pitch_deviation:.1f} deg", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # update simulated vehicle speed based on drowsiness state
    if drowsy:
        speed = max(0.0, speed - BRAKE_RATE)
    else:
        speed = min(MAX_SPEED, speed + RECOVER_RATE)

    draw_vehicle_panel(frame, speed, braking=drowsy)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()