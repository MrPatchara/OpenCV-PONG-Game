# 🎮 OpenCV PONG Game

<div align="center">

![OpenCV](https://img.shields.io/badge/OpenCV-5.0+-green.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A hand-controlled Pong game using OpenCV and MediaPipe hand tracking**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Game Modes](#-game-modes) • [Controls](#-controls)

</div>

---

## 📖 About

OpenCV PONG Game is an innovative take on the classic Pong arcade game, where players control paddles using **hand gestures** detected through a webcam. Built with OpenCV and MediaPipe, this project combines computer vision with interactive gaming for a unique, hands-free gaming experience.

### 🎯 Key Highlights

- ✨ **Hand Tracking Control** - Move your hands to control the paddles
- 🤖 **AI Opponent** - Play against an intelligent AI in single-player mode
- 👥 **Two-Player Mode** - Challenge a friend with dual hand tracking
- 📈 **Progressive Difficulty** - Game speed increases as you progress through stages
- 🎨 **Fullscreen Experience** - Immersive gameplay with beautiful graphics

---

## ✨ Features

### 🎮 Gameplay Features

- **Real-time Hand Detection** - Accurate hand tracking using MediaPipe
- **Dual Game Modes**:
  - 🆚 **VS AI Mode** - Play against an AI opponent
  - 👥 **VS Player Mode** - Two-player local multiplayer
- **Progressive Stages** - Difficulty increases every 5 points
- **Dynamic Ball Physics** - Realistic ball movement and collision detection
- **Score Tracking** - Real-time score display for both players
- **Game Over Screen** - Visual feedback when the game ends

### 🛠️ Technical Features

- **OpenCV Integration** - Real-time video processing and rendering
- **MediaPipe Hand Tracking** - State-of-the-art hand detection
- **Fullscreen Mode** - Immersive gaming experience
- **Webcam Integration** - Uses your default camera for hand tracking
- **Cross-Platform** - Works on Windows, macOS, and Linux

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam/Camera
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/OpenCV-PONG-Game.git
cd OpenCV-PONG-Game
```

### Step 2: Install Dependencies

```bash
pip install opencv-python cvzone numpy
```

Or install from requirements file (if available):

```bash
pip install -r requirements.txt
```

### Step 3: Verify Resources

Make sure the `Resources` folder contains all necessary images:
- `Background.png`
- `Ball.png`
- `bat1.png`
- `bat2.png`
- `gameOver.png`

---

## 🎮 Usage

### Running the Game

1. **Ensure your webcam is connected and working**

2. **Run the latest version**:
   ```bash
   python v5.py
   ```

   Or run other versions:
   ```bash
   python v3.py
   python v4.py
   ```

3. **Position yourself** in front of the camera so your hands are visible

4. **Select a game mode** from the main menu:
   - Press `1` for VS AI mode
   - Press `2` for VS Player mode
   - Press `ESC` to quit

5. **Start playing!** Move your hands up and down to control the paddles

---

## 🎯 Game Modes

### 🆚 VS AI Mode (Single Player)

- Control the **left paddle** with your **left hand**
- The **right paddle** is controlled by an AI opponent
- Perfect for solo gaming sessions

### 👥 VS Player Mode (Two Players)

- **Left player**: Control the left paddle with your **left hand**
- **Right player**: Control the right paddle with your **right hand**
- Great for competitive local multiplayer

---

## ⌨️ Controls

| Key | Action |
|-----|--------|
| `1` | Start VS AI mode |
| `2` | Start VS Player mode |
| `R` | Reset game (when game over) |
| `ESC` | Return to main menu / Quit |

### Hand Controls

- **Move your hand up/down** to control the paddle position
- The paddle follows your hand's vertical position
- Keep your hand visible to the camera for best tracking

---

## 📁 Project Structure

```
OpenCV-PONG-Game/
│
├── v5.py                 # Latest version (recommended)
├── v4.py                 # Previous version
├── v3.py                 # Previous version
├── v2.py                 # Previous version
├── game_v1.py            # Original version
│
├── Resources/            # Game assets
│   ├── Background.png    # Game background
│   ├── Ball.png          # Ball sprite
│   ├── bat1.png          # Left paddle
│   ├── bat2.png          # Right paddle
│   └── gameOver.png      # Game over screen
│
├── README.md             # This file
└── LICENSE               # MIT License
```

---

## 🎨 How It Works

1. **Camera Capture**: The game captures video from your webcam in real-time
2. **Hand Detection**: MediaPipe detects and tracks hand landmarks
3. **Paddle Control**: Hand position is mapped to paddle movement
4. **Ball Physics**: The ball moves with realistic physics and collision detection
5. **Score Tracking**: Points are awarded when the ball hits a paddle
6. **Stage Progression**: Game speed increases every 5 combined points

---

## 🔧 Requirements

### Python Packages

- `opencv-python` - Computer vision and image processing
- `cvzone` - Simplified OpenCV utilities and hand tracking
- `numpy` - Numerical operations

### System Requirements

- **OS**: Windows, macOS, or Linux
- **Camera**: Webcam or built-in camera
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **CPU**: Any modern processor

---

## 🐛 Troubleshooting

### Camera Not Detected

- Ensure your camera is connected and not being used by another application
- Try changing the camera index in the code: `cv2.VideoCapture(1)` instead of `cv2.VideoCapture(0)`

### Hand Not Detected

- Ensure good lighting conditions
- Keep your hands clearly visible to the camera
- Try adjusting the `detectionCon` parameter in the code (default: 0.8)

### Performance Issues

- Close other applications using the camera
- Reduce camera resolution in the code if needed
- Ensure you have sufficient lighting

### Missing Resources

- Verify all image files are in the `Resources/` folder
- Check that file names match exactly (case-sensitive)

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

- 🐛 Report bugs
- 💡 Suggest new features
- 🔧 Submit pull requests
- 📖 Improve documentation

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenCV** - For computer vision capabilities
- **MediaPipe** - For hand tracking technology
- **cvzone** - For simplified OpenCV utilities
- **Classic Pong** - For the original game concept

---

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/OpenCV-PONG-Game/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/OpenCV-PONG-Game/discussions)

---

<div align="center">

**Made with ❤️ using OpenCV and Python**

⭐ Star this repo if you find it interesting!

</div>
