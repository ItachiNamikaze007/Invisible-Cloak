# 🧙 Invisible Cloak — Real-Time OpenCV Demo

A **Harry Potter-style invisibility cloak** built with Python and OpenCV with **Multi-Color Selection** and **Interactive Color Calibration**.

---

## 📌 What It Does

The application captures your webcam feed in real time and makes any selected color cloth appear transparent — replacing it seamlessly with a pre-recorded background image.

Supports preset colors (**Red, Green, Blue, Yellow**) as well as an interactive **Custom Color Sampler** where you can click anywhere on your cloth to calibrate in real time.

---

## ⚙️ How It Works

### 1 — Color Selection & Calibration
At startup, choose your cloak color from a terminal menu:
- `1` → **Red** (uses dual-range HSV to cover hue wrap-around at 0° & 180°)
- `2` → **Green**
- `3` → **Blue**
- `4` → **Yellow**
- `5` → **Custom Color**: Opens an interactive window where you click your cloth. The app samples a patch, calculates HSV bounds automatically, and lets you confirm with `ENTER`/`SPACE`.

### 2 — Clean Background Capture
After selecting a color:
1. **Warning**: A banner asks you to step away.
2. **Countdown**: 3… 2… 1 countdown over live video.
3. **Capture**: Averages clean frames of the empty room.
4. **Ready**: Confirmation screen signals you can enter with your cloak.

### 3 — HSV Color Detection
Live frames are converted from **BGR → HSV**.
HSV separates *color (hue)* from *lightness (value)* and *purity (saturation)*, making color segmentation robust under varying illumination.

### 4 — Mask Cleanup & Edge Refinement
1. **Thresholding**: Multi-range HSV matching.
2. **Morphological Open** (3×3): Removes salt-and-pepper noise.
3. **Morphological Close** (7×7): Fills interior shadows/holes inside the cloak.
4. **Contour Area Filter**: Ignores small stray color specks (< 3000 px²).
5. **Erosion** (`MASK_ERODE_KERNEL`): Pulls boundary inward by ~1–2 px to remove edge fringe.
6. **Gaussian Feathering** (`MASK_FEATHER_KERNEL`): Softens the boundary into a smooth alpha gradient.

### 5 — Float Alpha Compositing
```
output = alpha * background + (1 - alpha) * live_frame
```
Uses floating-point alpha blending so feathered edges transition smoothly without red/colored borders or hard pixelation.

---

## 🛠️ Technologies Used

| Library | Purpose |
|---------|---------|
| **Python 3.8+** | Core language |
| **OpenCV (`cv2`)** | Webcam capture, HSV thresholding, morphology, mouse callbacks, alpha compositing |
| **NumPy** | Frame math, threshold arrays |

---

## 📦 Installation

> **Prerequisites:** Python 3.8 or newer.

```bash
# Navigate into the project folder
cd invisible_cloak

# (Recommended) Create a virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
python cloak.py
```

1. **Select Color**: Choose `1-5` from the terminal menu.
2. **Custom Calibration (if option 5 chosen)**:
   - Click on your cloth in the calibration window.
   - Press `ENTER` or `SPACE` to confirm.
3. **Step Away**: Watch the 3-second countdown and step out of frame.
4. **Step In**: Hold up your colored cloth to become invisible!

---

## ⌨️ Keyboard Controls

| Key | Context | Action |
|-----|---------|--------|
| `Q` | Main App | Quit application |
| `B` | Main App | Recapture clean background |
| `D` | Main App | Toggle **debug mask window** |
| `ENTER` / `SPACE` | Custom Calibration | Confirm sampled color |
| `ESC` | Custom Calibration | Cancel calibration |

---

## 🎨 Color Presets & Tuning

Preset bounds in `cloak.py`:

```python
COLOR_PRESETS = {
    "RED": [
        ((  0, 150, 100), ( 10, 255, 255)),
        ((165, 150, 100), (180, 255, 255))
    ],
    "GREEN":  [((35, 120, 100), ( 85, 255, 255))],
    "BLUE":   [((90, 120, 100), (135, 255, 255))],
    "YELLOW": [((20, 140, 140), ( 35, 255, 255))]
}
```

---

## 📁 Project Structure

```
invisible_cloak/
├── cloak.py          ← Main application with multi-color & calibration logic
├── requirements.txt  ← Python dependencies
└── README.md         ← Documentation
```

---

## 📜 License

Free to use and modify for educational and personal projects.
