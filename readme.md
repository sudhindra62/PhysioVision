# 🧠 PhysioTrack AI

> **An intelligent real-time physiotherapy monitoring and biomechanical correction system powered by Computer Vision, Pose Estimation, and Geometric Motion Analysis.**

PhysioTrack AI transforms traditional physiotherapy into a data-driven rehabilitation experience by automatically evaluating exercise form, tracking range of motion, and delivering real-time corrective feedback.

---

## 📌 Overview

PhysioTrack AI is a deterministic biomechanical evaluation engine designed for rehabilitation and physiotherapy exercises.

The system automatically converts reference exercise videos into mathematical motion profiles and JSON-based rule sets. During live sessions, user movements are compared against these calibrated baselines to evaluate posture quality, movement accuracy, range of motion, and exercise consistency.

---

## ✨ Key Features

- 🎯 Autonomous Exercise Calibration
- 📐 Real-Time Range of Motion (ROM) Analysis
- 🔄 Dynamic Time Warping (DTW) Form Matching
- ⚙️ Finite State Machine (FSM) Evaluation
- 🗣️ Real-Time Corrective Feedback
- 🧭 Advanced Alignment Validation
- 📊 Progress Tracking & Analytics

---

## 🏗️ System Architecture

Vision Engine (MediaPipe)
→ Geometry Engine
→ DTW Matcher
→ ROM Judge (FSM)
→ Analytics & SQLite

---

## 🧰 Tech Stack

| Technology   | Purpose              |
| ------------ | -------------------- |
| Python 3.11+ | Core logic           |
| MediaPipe    | Pose estimation      |
| OpenCV       | Video processing     |
| Streamlit    | Web UI               |
| WebRTC       | Browser streaming    |
| SQLite       | Persistence          |
| Pandas       | Analytics            |
| FastDTW      | Motion matching      |
| SciPy        | Numerical processing |

---

## 📂 Project Structure

```bash
PhysioTrack-AI/
├── core/
├── scripts/
├── assets/
├── data/
└── app.py
```

---

## 🚀 Getting Started

```bash
python -m venv venv
pip install -r requirements.txt
python scripts/extractor.py
python scripts/precompute_all.py
streamlit run app.py
```

---

## 👨‍💻 Authors

**Viral Bhoi**  
B.Tech Computer Engineering

**Rudra Chauhan**  
B.Tech Computer Engineering

---

## 🌟 Vision

Transforming rehabilitation through intelligent motion analysis and real-time biomechanical feedback.
