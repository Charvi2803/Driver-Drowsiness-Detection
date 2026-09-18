# Real-Time Driver Drowsiness Detection

A webcam-based ADAS-style system that detects driver drowsiness in real time using
facial landmark tracking, and demonstrates a simulated automatic emergency braking
response — showing the full detect → decide → act pipeline used in real driver-
monitoring systems.

![demo](assets/demo.gif)

## Why this project

Driver drowsiness is a major cause of road accidents, and real automotive ADAS
(Advanced Driver-Assistance Systems) use exactly this kind of vision-based
monitoring to catch it before it becomes dangerous. This project simulates that
pipeline end-to-end using nothing but a laptop webcam: live face tracking →
drowsiness detection via two independent signals → an alert → a simulated vehicle
response.

## How it works

1. **Face landmark tracking** — MediaPipe Face Mesh tracks 468 facial landmarks
   live from the webcam feed, including precise eye and head-pose reference points.

2. **Eye Aspect Ratio (EAR)** — calculated from 6 landmark points per eye using the
   standard formula from drowsiness-detection research (Soukupová & Čech, 2016).
   EAR drops sharply when eyes close. A consecutive-frame threshold (not a single
   frame) distinguishes genuine drowsiness from a normal blink.

3. **Head pose estimation** — using OpenCV's `solvePnP` with a standard 6-point 3D
   face model, the system calculates head pitch in real time. The system
   auto-calibrates a personal baseline pitch in the first ~1 second of runtime, then
   detects sustained deviation from that baseline as a "nodding off" signal —
   independent of eye closure, catching drowsiness patterns eye-tracking alone
   might miss.

4. **Alert system** — a visual on-screen warning plus an audio beep (played on a
   background thread so the video feed never freezes) fires the moment either
   signal crosses its threshold.

5. **Simulated vehicle response** — a live dashboard panel shows a simulated
   vehicle decelerating and displaying brake lights when drowsiness is detected,
   recovering once the driver is alert again. This demonstrates the full ADAS
   pipeline conceptually; it does not control a real vehicle.

## A real bug I hit and fixed

The head-pose pitch angle (from `cv2.decomposeProjectionMatrix`) doesn't center
around 0° for a normal forward-facing head — it centers around ~170-180° due to
how this specific angle decomposition is computed. My first version assumed 0°
was neutral, which caused constant false "HEAD NOD" alerts even when sitting
normally. The fix was auto-calibrating a personal baseline pitch value during the
first second of runtime, then detecting *deviation from that baseline* instead of
an assumed fixed value — which also makes the system more robust across different
users and camera setups.

Separately, MediaPipe recently overhauled its Python API (moving from the classic
`mp.solutions` interface to a newer "Tasks" API), which meant the latest MediaPipe
version silently broke all `mp.solutions.face_mesh`-based code. I pinned the
project to `mediapipe==0.10.14`, the last version with reliable classic-API
support, and had to install Python 3.11 specifically for this project since
MediaPipe hadn't yet published compatible wheels for the newest Python release.

## Project structure


## Running it

This project requires Python 3.11 (MediaPipe's classic API isn't available on the
newest Python releases yet).

```bash
pip install -r requirements.txt
python drowsiness_alert.py
```

Sit normally for the ~1 second calibration phase at startup, then the system is live.
Press 'q' to quit.

## Tech stack

Python, OpenCV, MediaPipe, NumPy

## License

MIT