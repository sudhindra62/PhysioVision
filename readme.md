# 🧠 PhysioTrack AI

> Real-time physiotherapy posture monitoring and exercise correction system using Computer Vision, AI, and pose estimation.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge\&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge\&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20Tracking-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge\&logo=streamlit)

---

# 📌 Overview

PhysioTrack AI is a real-time physiotherapy exercise monitoring system that uses computer vision and pose estimation to analyze human posture during exercises. The system captures webcam input, detects body landmarks, calculates joint angles, compares them with predefined ideal exercise postures, and provides live corrective feedback with similarity scoring.

This project aims to assist users and physiotherapists in ensuring exercises are performed correctly while reducing the need for constant manual supervision.

Based on your project specification document: 

---

# 🚀 Features

## ✅ Real-Time Pose Detection

* Detects full-body landmarks using MediaPipe Pose
* Live webcam tracking
* Skeleton overlay visualization

## 📐 Joint Angle Calculation

* Calculates:

  * Shoulder angles
  * Elbow angles
  * Hip angles
  * Knee angles

## 🎯 Similarity Scoring

* Compares user posture with ideal exercise posture
* Generates posture correctness percentage
* Tracks exercise accuracy in real time

## 🛠 Corrective Feedback

* Highlights incorrect joints
* Displays posture correction messages
* Provides live alignment notifications

## 📊 Session Analytics

* Stores exercise history locally
* Tracks improvement over time
* Displays progress charts and analytics

Project feature references:  

---

# 🏗 System Workflow

```text
Webcam Input
      ↓
Pose Detection
      ↓
Landmark Extraction
      ↓
Joint Angle Calculation
      ↓
Similarity Analysis
      ↓
Corrective Feedback
      ↓
Analytics & History
```

Workflow based on SRS description: 

---

# 🧰 Tech Stack

| Technology | Purpose                   |
| ---------- | ------------------------- |
| Python     | Core development          |
| OpenCV     | Video frame processing    |
| MediaPipe  | Pose estimation           |
| Streamlit  | Frontend GUI              |
| NumPy      | Mathematical computations |
| SQLite     | Local session storage     |
| Matplotlib | Analytics visualization   |

Software interface references: 

---

# 📂 Project Structure

```bash
PhysioTrack-AI/
│
├── app.py                     # Main Streamlit application
├── requirements.txt           # Project dependencies
├── exercises/                 # Ideal posture datasets
├── models/                    # Pose analysis logic
├── utils/                     # Utility functions
├── analytics/                 # Session analytics & charts
├── database/                  # SQLite storage
├── assets/                    # Images/icons/UI assets
├── README.md
└── demo/
    └── sample_output.mp4
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/PhysioTrack-AI.git
cd PhysioTrack-AI
```

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

# 🧍 Supported Functionalities

* Real-time webcam capture
* Full-body landmark detection
* Pose similarity analysis
* Multiple physiotherapy exercises
* Live correction system
* Session history tracking
* Performance analytics dashboard

Functional requirement references: 

---

# 💡 Future Enhancements

* Multi-user support
* Cloud synchronization
* Therapist dashboard
* AI-based personalized exercise recommendations
* Mobile application support
* 3D pose estimation
* Voice-assisted corrections

Future scope inspired from document constraints and extensions:  

---

# 🖥 Operating Requirements

## Hardware

* Webcam
* Minimum 8GB RAM

## Software

* Windows 10/11
* Python 3.11+
* Streamlit
* OpenCV
* MediaPipe

Operating environment reference: 

---

# 📈 Performance Requirements

* Minimum 15 FPS
* Response time below 200ms
* Real-time posture correction

Performance references: 

---

# 🔒 Security & Privacy

* Local-only data storage
* No external data sharing
* Secure session handling

Security references: 

---

# 🤝 Contributing

Contributions are welcome! 🚀

```bash
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request
```

---



# 👨‍💻 Author

** Viral Bhoi and Rudra Chauhan**
B.Tech Computer Engineering Students | Competitive Programmers | AI & Computer Vision Enthusiast

---

# 🌟 Vision

Turning physiotherapy into an intelligent real-time interactive experience where cameras become posture guardians and algorithms whisper corrections before injuries can speak. 🦾
