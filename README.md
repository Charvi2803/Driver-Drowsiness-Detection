# Real-Time Driver Drowsiness Detection

A webcam-based, ADAS-style system that detects driver drowsiness in real time using facial landmark tracking, and demonstrates a simulated automatic emergency braking response. It shows the full **detect → decide → act** pipeline used in real driver-monitoring systems.

## Why this project

Driver drowsiness is a major cause of road accidents, and automotive ADAS (Advanced Driver-Assistance Systems) use exactly this kind of vision-based monitoring to catch it before it becomes dangerous. This project simulates that pipeline end-to-end using nothing but a laptop webcam: live face tracking → drowsiness detection from two independent signals → an alert → a simulated vehicle response.

## How it works

1. **Face landmark tracking:** MediaPipe Face Mesh tracks 468 facial landmarks live from the webcam feed, including precise eye and head-pose reference points.
2. **Eye Aspect Ratio (EAR):** calculated from 6 landmark points per eye using the standard formula from drowsiness-detection research (Soukupová & Čech, 2016). EAR drops sharply when the eyes close. The eyes must stay closed for a sustained period (1 second by default), not a single frame, which separates genuine drowsiness from a normal blink.
3. **Head pose estimation:** OpenCV's `solvePnP` with a standard 6-point 3D face model gives the head pitch in real time. The system calibrates a personal baseline pitch over the first ~30 frames, then treats a sustained deviation from that baseline as a "nodding off" signal. This works independently of eye closure and catches drowsiness patterns that eye tracking alone can miss. Angle differences are wrap-safe (±180°).
4. **Alert system:** an on-screen warning plus an audio beep fires when either signal persists past the threshold. The beep plays on a background thread so the video feed never freezes. It is cross-platform: `winsound` on Windows, a terminal bell on macOS/Linux.
5. **Simulated vehicle response:** a live dashboard panel shows a simulated vehicle decelerating with brake lights on when drowsiness is detected, then recovering once the driver is alert again. This demonstrates the ADAS pipeline conceptually; it does not control a real vehicle.
6. **Robustness:**
   - Thresholds are time-based, so behaviour is the same at any webcam frame rate.
   - Timers reset when the face leaves the frame, and a "No face detected" notice is shown.
   - A **vehicle moving/stopped** toggle pauses monitoring while stopped to avoid false alarms, and the head-pose baseline is recalibrated automatically when driving resumes.

## Controls

| Key | Action |
|-----|--------|
| `m` | Toggle vehicle moving / stopped (monitoring pauses while stopped) |
| `r` | Recalibrate the head-pose baseline |
| `q` | Quit |

Click the video window first so it receives the keypresses.

## Setup and run

Requires Python 3.11 and a webcam.

```bash
git clone https://github.com/Charvi2803/Driver-Drowsiness-Detection.git
cd Driver-Drowsiness-Detection
pip install -r requirements.txt
python drowsiness_alert.py
```

On Windows you can also use `py -3.11 drowsiness_alert.py`. When the window opens, sit in your normal driving posture for about a second while it calibrates.

## Configuration

Thresholds are constants at the top of `drowsiness_alert.py`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `EAR_THRESHOLD` | 0.25 | EAR below this counts as eyes closed |
| `PITCH_DEVIATION_DEG` | 15 | Head pitch deviation from baseline counted as a nod |
| `ALERT_SECONDS` | 1.0 | How long a signal must persist before the alert fires |
| `CALIBRATION_FRAMES` | 30 | Frames used to calibrate the baseline pitch |

## Tech stack

Python · OpenCV · MediaPipe Face Mesh · NumPy

## Limitations

- Uses a generic 3D face model, so head-pitch estimates are approximate and rely on per-user calibration.
- EAR thresholds can vary between individuals and lighting conditions.
- The vehicle response is a simulation only.

## Reference

Soukupová, T. & Čech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks.* 21st Computer Vision Winter Workshop.