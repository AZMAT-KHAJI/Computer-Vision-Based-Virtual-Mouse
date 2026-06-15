<div align="center">

# 🖱️ Computer Vision–Based Virtual Mouse

**Control your computer with nothing but your hand.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange?style=flat-square)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)

A real-time, touchless human-computer interaction system powered by hand gesture recognition. Wave, pinch, and point your way through your desktop - no hardware required beyond a standard webcam.

</div>

---

## ✨ Features

- **Real-time hand tracking** via MediaPipe landmark detection
- **Smooth cursor movement** using index finger position
- **Gesture-based actions** — click, right-click, double-click, drag, and scroll
- **FPS monitoring** for live performance feedback
- **Optional KNN model** for trained gesture classification

---

## 🖐️ Supported Gestures

| Gesture | Action |
|---|---|
| Index finger up | Move cursor |
| Thumb–index pinch | Left click |
| Thumb + index + middle | Right click |
| Double pinch | Double click |
| Pinch & hold | Drag and drop |
| Two-finger gesture | Scroll up / down |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Hand detection | MediaPipe |
| Image processing | OpenCV |
| Mouse automation | PyAutoGUI |
| Gesture classification | Scikit-learn (KNN) |
| Data handling | NumPy, Pandas |

---

## ⚙️ System Workflow

```
Webcam Feed
    │
    ▼
MediaPipe Hand Landmark Detection
    │
    ▼
Finger Position & Feature Extraction
    │
    ▼
Gesture Recognition
(Rule-based logic  ──or──  KNN Classifier)
    │
    ▼
Mouse Action Mapping (PyAutoGUI)
    │
    ▼
System Mouse Event
```

---

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/AZMAT-KHAJI/Computer-Vision-Based-Virtual-Mouse.git
cd Computer-Vision-Based-Virtual-Mouse
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
python hand_mouse.py
```

> Make sure your webcam is connected and accessible before launching.

---

## 📁 Project Structure

```
Computer-Vision-Based-Virtual-Mouse/
│
├── hand_mouse.py                  # Application entry point
├── requirements.txt         # Python dependencies
├── README.md
```

---

## 💡 Use Cases

- **Accessibility** — hands-free control for users with mobility limitations
- **Touchless workstations** — hygiene-critical or sterile environments
- **HCI research** — gesture-based interface experimentation
- **Smart presentations** — navigate slides without a clicker

---

## 🔭 Future Enhancements

- [ ] Multi-hand gesture support
- [ ] Custom gesture training UI
- [ ] Deep learning-based classification (CNN / Transformer)
- [ ] Volume and brightness control
- [ ] Virtual keyboard integration
- [ ] Cross-platform optimization (macOS, Linux, Windows)

---

## 👤 Developer

**Azmat Khaji**
GitHub: [@AZMAT-KHAJI](https://github.com/AZMAT-KHAJI)

---

<div align="center">

*Built with computer vision and ✋ a lot of hand-waving.*

</div>
