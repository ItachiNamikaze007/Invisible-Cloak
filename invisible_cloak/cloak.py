# -*- coding: utf-8 -*-
"""
Invisible Cloak - Real-Time OpenCV Demo
========================================
A Harry Potter-style invisibility effect using your webcam with multi-color support.

HOW IT WORKS (brief)
---------------------
1. Select cloak color preset (RED, GREEN, BLUE, YELLOW) or calibrate a CUSTOM color by clicking.
2. Step away while a clean background is captured.
3. Each live frame is converted to HSV color space.
4. Pixels matching the selected cloak color range are masked out and filtered.
5. Those pixels are blended with the stored background using soft alpha compositing.

Keyboard Controls
-----------------
  Q  ->  Quit
  B  ->  Recapture background
  D  ->  Toggle debug mask window
"""

import sys
import time

import cv2
import numpy as np


# =============================================================================
#  COLOR PRESETS & TUNABLE PARAMETERS
# =============================================================================

# Each preset maps to a list of (LOWER_HSV, UPPER_HSV) tuples.
# Red requires two ranges because hue wraps around 0/180.
COLOR_PRESETS = {
    "RED": [
        ((  0, 150, 100), ( 10, 255, 255)),
        ((165, 150, 100), (180, 255, 255))
    ],
    "GREEN": [
        (( 35, 120, 100), ( 85, 255, 255))
    ],
    "BLUE": [
        (( 90, 120, 100), (135, 255, 255))
    ],
    "YELLOW": [
        (( 20, 140, 140), ( 35, 255, 255))
    ]
}

# --- Contour filter ---
# Blobs smaller than this area (in pixels) are treated as noise and ignored.
# At 640x480 a value of 3000 corresponds to roughly a 55x55 px square.
MIN_CONTOUR_AREA = 3000

# --- Mask edge refinement ---
# After contour filtering the mask boundary sits on top of cloak pixels,
# causing a visible colored outline in the final composite.
# Eroding pulls the boundary inward so those edge pixels are removed,
# and Gaussian blur feathers the transition for a natural, soft edge.
MASK_ERODE_KERNEL     = 3    # px  — small 3x3 removes ~1-2 px border
MASK_ERODE_ITERATIONS = 1    # 1 iteration keeps erosion conservative
MASK_FEATHER_KERNEL   = 21   # px  — blur spread; must be odd

# --- Background capture ---
BG_CLEAN_FRAMES   = 15    # frames averaged after countdown ends
BG_COUNTDOWN_SECS = 3     # seconds for user to step out of frame


# =============================================================================
#  HELPER: Safe window check & destruction
# =============================================================================

def _window_exists(name: str) -> bool:
    """
    Return True if the named OpenCV window currently exists.
    cv2.getWindowProperty() returns -1.0 when closed or non-existent.
    """
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def _safe_destroy(name: str) -> None:
    """Destroy a window safely if it currently exists."""
    if _window_exists(name):
        cv2.destroyWindow(name)


# =============================================================================
#  CUSTOM COLOR CALIBRATION
# =============================================================================

def get_color_ranges_from_sample(h: float, s: float, v: float) -> list:
    """
    Compute lower & upper HSV bounds around a sampled HSV value.
    Enforces minimum saturation/value floors to prevent gray/skin false positives.
    Handles hue wrap-around if hue +/- tolerance crosses 0 or 180.
    """
    h_tol = 12
    lower_s = max(int(s) - 60, 100)
    upper_s = 255
    lower_v = max(int(v) - 60, 80)
    upper_v = 255

    h_min = h - h_tol
    h_max = h + h_tol

    ranges = []
    if h_min < 0:
        ranges.append(((0, lower_s, lower_v), (int(h_max), upper_s, upper_v)))
        ranges.append(((int(h_min + 180), lower_s, lower_v), (180, upper_s, upper_v)))
    elif h_max > 180:
        ranges.append(((int(h_min), lower_s, lower_v), (180, upper_s, upper_v)))
        ranges.append(((0, lower_s, lower_v), (int(h_max - 180), upper_s, upper_v)))
    else:
        ranges.append(((int(h_min), lower_s, lower_v), (int(h_max), upper_s, upper_v)))

    return ranges


def calibrate_custom_color(cap: cv2.VideoCapture) -> list:
    """
    Interactive color calibration window.
    User clicks on their cloth to sample its HSV color.
    Press ENTER / SPACE to confirm, ESC to cancel.
    """
    CALIB_WIN = "Custom Color Calibration — Click Cloth"
    clicked_pos = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal clicked_pos
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_pos = (x, y)

    cv2.namedWindow(CALIB_WIN)
    cv2.setMouseCallback(CALIB_WIN, mouse_callback)

    sampled_hsv = None
    sampled_ranges = None

    print("[INFO] Opening calibration window. Click on your cloth to sample color.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        display = frame.copy()
        h, w = frame.shape[:2]

        if clicked_pos is not None:
            cx, cy = clicked_pos
            # Sample a 7x7 patch around click location
            x1, x2 = max(0, cx - 3), min(w, cx + 4)
            y1, y2 = max(0, cy - 3), min(h, cy + 4)
            patch = hsv_frame[y1:y2, x1:x2]

            if patch.size > 0:
                mean_val = np.mean(patch, axis=(0, 1))
                sampled_hsv = (float(mean_val[0]), float(mean_val[1]), float(mean_val[2]))
                sampled_ranges = get_color_ranges_from_sample(*sampled_hsv)

                # Draw indicator & color info
                cv2.circle(display, (cx, cy), 8, (0, 255, 0), 2)
                cv2.circle(display, (cx, cy), 2, (0, 0, 255), -1)

                info_str = f"Sampled HSV: H={int(sampled_hsv[0])} S={int(sampled_hsv[1])} V={int(sampled_hsv[2])}"
                cv2.putText(display, info_str, (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(display, "Press ENTER / SPACE to confirm selection", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        # Header instructions
        cv2.rectangle(display, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(display, "CLICK ON YOUR CLOTH TO SAMPLE COLOR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, "ENTER/SPACE = Confirm  |  ESC = Cancel", (w - 380, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(CALIB_WIN, display)

        if not _window_exists(CALIB_WIN):
            print("[INFO] Calibration window closed.")
            break

        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):  # ENTER or SPACE
            if sampled_ranges is not None:
                print(f"[INFO] Custom color selected: HSV ~ {tuple(int(x) for x in sampled_hsv)}")
                _safe_destroy(CALIB_WIN)
                return sampled_ranges
            else:
                print("[WARNING] Please click on your cloth first to sample a color!")
        elif key == 27:  # ESC
            print("[INFO] Calibration cancelled by user.")
            break

    _safe_destroy(CALIB_WIN)
    print("[INFO] Defaulting to RED preset.")
    return COLOR_PRESETS["RED"]


def select_cloak_color(cap: cv2.VideoCapture) -> tuple[str, list]:
    """
    Present terminal menu for color selection.
    Returns (color_name, list_of_hsv_ranges).
    """
    print("\n" + "=" * 58)
    print("  Invisible Cloak — Select Cloak Color")
    print("=" * 58)
    print("  1 -> Red      (Default, dual-range HSV)")
    print("  2 -> Green")
    print("  3 -> Blue")
    print("  4 -> Yellow")
    print("  5 -> Custom color (Interactive click-to-sample)")
    print("=" * 58)

    while True:
        try:
            choice = input("Enter choice (1-5) [default: 1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Selection interrupted. Using RED.")
            return "RED", COLOR_PRESETS["RED"]

        if choice in ("", "1"):
            return "RED", COLOR_PRESETS["RED"]
        elif choice == "2":
            return "GREEN", COLOR_PRESETS["GREEN"]
        elif choice == "3":
            return "BLUE", COLOR_PRESETS["BLUE"]
        elif choice == "4":
            return "YELLOW", COLOR_PRESETS["YELLOW"]
        elif choice == "5":
            ranges = calibrate_custom_color(cap)
            return "CUSTOM", ranges
        else:
            print("[ERROR] Invalid choice. Please enter a number between 1 and 5.")


# =============================================================================
#  BACKGROUND CAPTURE
# =============================================================================

def capture_background(cap: cv2.VideoCapture) -> np.ndarray:
    """
    Capture a CLEAN background — one that does NOT contain the user.
    Uses countdown workflow: 1s warning -> 3s countdown -> capture -> ready.
    """
    MAIN_WIN = "Invisible Cloak"
    print("[INFO] Background capture starting — please step out of frame.")

    # Phase 1: Warning
    phase1_end = time.time() + 1.0
    while time.time() < phase1_end:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frame = cv2.flip(frame, 1)

        overlay = frame.copy()
        banner = overlay[0:80, :]
        banner[:] = (banner * 0.4).astype(np.uint8)

        cv2.putText(overlay, "STEP AWAY FROM CAMERA", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, "Capturing background in 3 seconds ...", (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(MAIN_WIN, overlay)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quit requested.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    # Phase 2: Countdown 3 ... 2 ... 1
    for count in range(BG_COUNTDOWN_SECS, 0, -1):
        digit_end = time.time() + 1.0
        while time.time() < digit_end:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frame = cv2.flip(frame, 1)

            overlay = frame.copy()
            h, w = overlay.shape[:2]

            label = str(count)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 6, 8)
            tx = (w - tw) // 2
            ty = (h + th) // 2

            cv2.putText(overlay, label, (tx + 3, ty + 3), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 0), 10, cv2.LINE_AA)
            cv2.putText(overlay, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 255), 8, cv2.LINE_AA)
            cv2.putText(overlay, "STEP AWAY FROM CAMERA", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(MAIN_WIN, overlay)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Quit requested.")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)

    # Phase 3: Capture clean frames
    print("[INFO] Countdown complete — capturing clean background ...")
    accumulator: np.ndarray | None = None
    captured = 0

    while captured < BG_CLEAN_FRAMES:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frame = cv2.flip(frame, 1)

        f64 = frame.astype(np.float64)
        accumulator = f64 if accumulator is None else accumulator + f64
        captured += 1

        h, w = frame.shape[:2]
        progress_img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(progress_img, f"Capturing background ... {captured}/{BG_CLEAN_FRAMES}", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(MAIN_WIN, progress_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quit requested.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    background = (accumulator / BG_CLEAN_FRAMES).astype(np.uint8)

    # Phase 4: Confirmation
    print("[INFO] Background captured successfully — user may re-enter frame.")
    ready_end = time.time() + 1.5
    while time.time() < ready_end:
        confirm = background.copy()
        confirm[0:90, :] = (confirm[0:90, :] * 0.35).astype(np.uint8)
        cv2.putText(confirm, "BACKGROUND READY", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 100), 2, cv2.LINE_AA)
        cv2.putText(confirm, "Step in with your cloth!", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(MAIN_WIN, confirm)
        cv2.waitKey(30)

    return background


# =============================================================================
#  MASK CREATION
# =============================================================================

def create_cloak_mask(hsv_frame: np.ndarray, color_ranges: list) -> np.ndarray:
    """
    Build a mask highlighting pixels in the specified color_ranges.

    Parameters
    ----------
    hsv_frame    : HSV image frame
    color_ranges : list of ((lower_hsv), (upper_hsv)) tuples

    Pipeline
    --------
    1. HSV thresholding (ORing all ranges together)
    2. Morphological OPEN  (3x3 kernel, 1 iter)
    3. Morphological CLOSE (7x7 kernel, 2 iters)
    4. Contour filter     (area >= MIN_CONTOUR_AREA)
    5. Mask erosion       (MASK_ERODE_KERNEL, MASK_ERODE_ITERATIONS)
    6. Gaussian feather   (MASK_FEATHER_KERNEL)
    """
    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)

    # Step 1 — Color threshold: OR all specified HSV ranges
    for lo, hi in color_ranges:
        lo_arr = np.array(lo, dtype=np.uint8)
        hi_arr = np.array(hi, dtype=np.uint8)
        r_mask = cv2.inRange(hsv_frame, lo_arr, hi_arr)
        mask = cv2.bitwise_or(mask, r_mask)

    # Step 2 — Open: remove isolated noise pixels
    kernel_open = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Step 3 — Close: fill small holes / shadows inside cloak
    kernel_close = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Step 4 — Contour filter: keep only blobs >= MIN_CONTOUR_AREA
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA:
            cv2.drawContours(clean_mask, [cnt], -1, 255, cv2.FILLED)

    # Step 5 — Erode: pull mask boundary inward to eliminate edge color fringe
    erode_k = np.ones((MASK_ERODE_KERNEL, MASK_ERODE_KERNEL), np.uint8)
    clean_mask = cv2.erode(clean_mask, erode_k, iterations=MASK_ERODE_ITERATIONS)

    # Step 6 — Feather: Gaussian blur hard binary edge into soft gradient
    fk = MASK_FEATHER_KERNEL if MASK_FEATHER_KERNEL % 2 == 1 else MASK_FEATHER_KERNEL + 1
    clean_mask = cv2.GaussianBlur(clean_mask, (fk, fk), 0)

    return clean_mask


# =============================================================================
#  INVISIBILITY COMPOSITING
# =============================================================================

def apply_invisibility(
    frame: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Replace cloak pixels in live frame with background pixels using float alpha blending.
    """
    alpha = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

    frame_f = frame.astype(np.float32)
    bg_f    = background.astype(np.float32)
    blended = alpha * bg_f + (1.0 - alpha) * frame_f

    return np.clip(blended, 0, 255).astype(np.uint8)


# =============================================================================
#  MAIN APPLICATION
# =============================================================================

DEBUG_WINDOW = "Debug — Cloak Mask"

def main() -> None:
    # 1. Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(
            "[ERROR] Cannot open webcam.\n"
            "  - Check that the camera is not in use by another app.\n"
            "  - Try cv2.VideoCapture(1) if you have multiple cameras."
        )
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 2. Select cloak color mode
    color_name, color_ranges = select_cloak_color(cap)
    print(f"[INFO] Active Cloak Color: {color_name}")

    # 3. Capture clean background
    background = capture_background(cap)

    # 4. Main loop
    fps_start   = time.time()
    frame_count = 0
    show_debug  = False

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARNING] Dropped frame — retrying ...")
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        frame_count += 1

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = create_cloak_mask(hsv, color_ranges)
        output = apply_invisibility(frame, background, mask)

        # Overlay info
        elapsed = time.time() - fps_start
        fps = frame_count / elapsed if elapsed > 0 else 0.0
        cv2.putText(
            output,
            f"COLOR: {color_name} | FPS: {fps:.1f} | Q=Quit B=Recapture D=Debug",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 255, 0), 2, cv2.LINE_AA,
        )

        cv2.imshow("Invisible Cloak", output)

        # Debug window
        if show_debug:
            debug_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(
                debug_view,
                f"MASK ({color_name} - white = cloak detected)",
                (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.imshow(DEBUG_WINDOW, debug_view)

            if not _window_exists(DEBUG_WINDOW):
                show_debug = False
        else:
            _safe_destroy(DEBUG_WINDOW)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[INFO] Quit — goodbye!")
            break

        elif key == ord("b"):
            _safe_destroy(DEBUG_WINDOW)
            show_debug = False
            print("[INFO] Recapturing background ...")
            background  = capture_background(cap)
            fps_start   = time.time()
            frame_count = 0

        elif key == ord("d"):
            show_debug = not show_debug
            if not show_debug:
                _safe_destroy(DEBUG_WINDOW)
            print(f"[INFO] Debug mask: {'ON' if show_debug else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
