"""
tracking.py — AI-Assisted Closed-Loop Virtual Camera Tracking System for SIH26169
Includes Interactive OpenCV Control Panel GUI for Real-Time Disturbance Tuning:
  - Atmospheric Disturbance (Clear, Haze, Fog, Rain, Low Light) & Severity Slider (0-100%)
  - Image Noise (None, Salt & Pepper, Gaussian, Poisson) & Level/Std Slider (0-20 px)
  - Camera Jitter Slider (+/- 0 to 20 px/frame)
  - Platform Carrier Drift Slider (+/- 0 to 20 px/frame)
  - Pan/Tilt Hardware Speed Slider (5.0 to 10.0 deg/s)
  - Target Motion: Horizontal, Straight Line, Circular, Figure-of-8, Random
  - 2D Kalman Filter & Morphological Top-Hat Filter for robust centroid lock
  - Auto-generated performance log CSV (Mandatory Deliverable 5)
"""

import math
import time
import csv
import os
import cv2
import numpy as np
from step2_camera import SimulatedCameraSource, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, build_debug_view
from step4_search import SpiralScanner

# ---- Benchmark & Operational Specifications ----
FPS = 30
SETPOINT_X = VIEWPORT_WIDTH // 2   # 320 px (optical center)
SETPOINT_Y = VIEWPORT_HEIGHT // 2  # 240 px (optical center)
SPEC_TRACKING_ERROR_PX = 10.0      # Spec constraint: tracking error <= 10 px (PDF Sr. No. 17)
SPEC_ACQ_TIME_SEC = 2.0            # Spec constraint: acquisition time <= 2.0s (PDF Sr. No. 16)
LOST_FRAME_LIMIT = 15              # 0.5s loss threshold before re-search

ATMOSPHERE_MODES = ["clear", "haze", "fog", "rain", "low_light"]
NOISE_MODES = ["none", "salt_pepper", "gaussian", "poisson"]
MOTION_MODES = ["horizontal", "straight_line", "circular", "figure_8", "random"]
MOTION_NAMES = {
    ord('1'): "horizontal",
    ord('2'): "straight_line",
    ord('3'): "circular",
    ord('4'): "figure_8",
    ord('5'): "random"
}


def create_control_panel():
    """
    Creates an interactive OpenCV Control Panel GUI window with trackbars
    for real-time tuning of atmospheric turbulence, sensor noise, jitter,
    platform carrier drift, pan speed, and target trajectory.
    Short, clear trackbar names prevent truncation in OpenCV window headers.
    """
    cv2.namedWindow("Control Panel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Control Panel", 540, 420)

    # 1. Atmospheric Disturbance (0=Clear, 1=Haze, 2=Fog, 3=Rain, 4=Low Light)
    cv2.createTrackbar("Atmos Mode (0-4)", "Control Panel", 0, 4, lambda x: None)
    # 2. Atmospheric Turbulence / Degradation Severity (0 to 100%)
    cv2.createTrackbar("Atmos Sev % (0-100)", "Control Panel", 75, 100, lambda x: None)
    # 3. Image Noise Type (0=None, 1=Salt & Pepper, 2=Gaussian, 3=Poisson)
    cv2.createTrackbar("Noise Mode (0-3)", "Control Panel", 0, 3, lambda x: None)
    # 4. Noise Level / Std Dev (0 to 20 px per spec)
    cv2.createTrackbar("Noise Lvl (0-20px)", "Control Panel", 10, 20, lambda x: None)
    # 5. Camera Jitter (+/- 0 to 20 px/frame per spec)
    cv2.createTrackbar("Jitter (+/-0-20px)", "Control Panel", 0, 20, lambda x: None)
    # 6. Platform Carrier Drift (+/- 0 to 20 px/frame per spec)
    cv2.createTrackbar("Drift (+/-0-20px)", "Control Panel", 0, 20, lambda x: None)
    # 7. Max Pan/Tilt Speed (5.0 to 10.0 deg/s per spec, scale x10)
    cv2.createTrackbar("Pan Spd x10 (50-100)", "Control Panel", 80, 100, lambda x: None)
    # 8. Target Motion Pattern (0=Horizontal, 1=Straight, 2=Circular, 3=Fig8, 4=Random)
    cv2.createTrackbar("Motion (0-4)", "Control Panel", 0, 4, lambda x: None)


def read_control_panel():
    """Reads all current slider positions from the Control Panel GUI."""
    atmos_idx = cv2.getTrackbarPos("Atmos Mode (0-4)", "Control Panel")
    atmos_sev = cv2.getTrackbarPos("Atmos Sev % (0-100)", "Control Panel")
    noise_idx = cv2.getTrackbarPos("Noise Mode (0-3)", "Control Panel")
    noise_lvl = cv2.getTrackbarPos("Noise Lvl (0-20px)", "Control Panel")
    jitter_px = cv2.getTrackbarPos("Jitter (+/-0-20px)", "Control Panel")
    plat_px = cv2.getTrackbarPos("Drift (+/-0-20px)", "Control Panel")
    pan_speed_x10 = max(50, cv2.getTrackbarPos("Pan Spd x10 (50-100)", "Control Panel"))
    motion_idx = cv2.getTrackbarPos("Motion (0-4)", "Control Panel")

    return (
        ATMOSPHERE_MODES[min(atmos_idx, 4)],
        atmos_sev / 100.0,
        NOISE_MODES[min(noise_idx, 3)],
        float(noise_lvl),
        float(jitter_px),
        float(plat_px),
        pan_speed_x10 / 10.0,
        MOTION_MODES[min(motion_idx, 4)]
    )


def draw_control_panel_status(atmos_mode, atmos_sev, noise_mode, noise_lvl,
                              jitter_px, plat_px, pan_deg, motion_mode):
    """Renders informative status card within the Control Panel GUI window."""
    canvas = np.full((240, 540, 3), 32, dtype=np.uint8)

    cv2.putText(canvas, "ISRO FSOC OPTICAL DISTURBANCE PANEL", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 2)
    cv2.line(canvas, (15, 33), (525, 33), (80, 80, 80), 1)

    # Atmosphere & Noise parameters
    cv2.putText(canvas, f"Atmosphere: {atmos_mode.upper()}  |  Severity: {int(atmos_sev*100)}%",
                (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1)
    cv2.putText(canvas, f"Image Noise: {noise_mode.upper()}  |  Intensity/Std: {int(noise_lvl)} px",
                (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1)

    # Mechanical vibrations & Carrier motion
    cv2.putText(canvas, f"Camera Jitter: +/-{int(jitter_px)} px/frame (Spec <= 20px)",
                (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 240, 200), 1)
    cv2.putText(canvas, f"Platform Motion: +/-{int(plat_px)} px/frame (Spec <= 20px)",
                (15, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 240, 200), 1)

    # Motion & Pan speed
    cv2.putText(canvas, f"Pan/Tilt Speed: {pan_deg:.1f} deg/s  |  Pattern: {motion_mode.upper()}",
                (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 200, 100), 1)

    # Keyboard hotkey reminder
    cv2.line(canvas, (15, 175), (525, 175), (80, 80, 80), 1)
    cv2.putText(canvas, "Keys: [1]-[5] Motion (1=Horizontal)  [SPACE] Pause  [R] Reset  [S] Save Log",
                (15, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1)

    return canvas


class RobustBeaconDetector:
    """
    Morphological and statistical beacon detector designed to withstand
    severe optical disturbances (10% Salt & Pepper, heavy Fog/Haze, low contrast).
    """
    def __init__(self, min_area=8, max_area=1500):
        self.min_area = min_area
        self.max_area = max_area
        self.tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def detect(self, frame):
        filtered = cv2.medianBlur(frame, 5)
        tophat = cv2.morphologyEx(filtered, cv2.MORPH_TOPHAT, self.tophat_kernel)
        min_val, max_val, _, _ = cv2.minMaxLoc(tophat)

        if max_val < 20:
            thresh_val = max(int(np.percentile(filtered, 99.8)), 25)
            _, binary = cv2.threshold(filtered, thresh_val, 255, cv2.THRESH_BINARY)
        else:
            thresh_val = max(int(max_val * 0.40), 18)
            _, binary = cv2.threshold(tophat, thresh_val, 255, cv2.THRESH_BINARY)

        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.open_kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    candidates.append((area, cx, cy))

        if candidates:
            candidates.sort(key=lambda item: abs(item[0] - 100))
            best_area, best_cx, best_cy = candidates[0]
            return best_cx, best_cy, best_area

        return None, None, 0


class KalmanBeaconTracker2D:
    """
    2D Constant Velocity (CV) Kalman Filter for coarse optical beacon tracking.
    State: [x, y, vx, vy]^T.
    Smooths high-frequency camera jitter (+/-20 px/frame) and provides predictive
    coasting during brief visual dropouts to ensure Re-acquisition Time <= 1.0s.
    """
    def __init__(self, dt=1.0 / FPS):
        self.dt = dt
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)

        self.Q = np.diag([0.5, 0.5, 2.0, 2.0]).astype(np.float32)
        self.R = np.eye(2, dtype=np.float32) * (10.0**2)
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 50.0
        self.initialized = False

    def reset(self):
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 50.0
        self.initialized = False

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, meas_x, meas_y):
        z = np.array([[meas_x], [meas_y]], dtype=np.float32)
        if not self.initialized:
            self.x = np.array([[meas_x], [meas_y], [0.0], [0.0]], dtype=np.float32)
            self.initialized = True
            return meas_x, meas_y

        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(4, dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return float(self.x[0, 0]), float(self.x[1, 0])


class PIDController2D:
    def __init__(self, kp=0.58, ki=0.035, kd=0.11):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.initialized = False

    def reset(self):
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.initialized = False

    def compute(self, error_x, error_y, max_speed_px):
        if not self.initialized:
            deriv_x = 0.0
            deriv_y = 0.0
            self.initialized = True
        else:
            deriv_x = error_x - self.prev_error_x
            deriv_y = error_y - self.prev_error_y

        self.prev_error_x = error_x
        self.prev_error_y = error_y

        cmd_x = (self.kp * error_x) + (self.ki * self.integral_x) + (self.kd * deriv_x)
        cmd_y = (self.kp * error_y) + (self.ki * self.integral_y) + (self.kd * deriv_y)

        cmd_magnitude = math.hypot(cmd_x, cmd_y)
        saturated = False

        if cmd_magnitude > max_speed_px and cmd_magnitude > 0:
            scale = max_speed_px / cmd_magnitude
            cmd_x *= scale
            cmd_y *= scale
            saturated = True

        if not saturated:
            self.integral_x += error_x
            self.integral_y += error_y

        return cmd_x, cmd_y


def compute_max_speed_px(pan_speed_deg, fov_deg, fps=FPS):
    px_per_deg = VIEWPORT_WIDTH / fov_deg
    return (pan_speed_deg / fps) * px_per_deg


def render_dashboard(frame, state, motion_type, acq_time, search_frames,
                     error_dist, avg_error, lock_rate, cx, cy, kf_x, kf_y,
                     pan_deg, fov_deg, current_fps, noise_mode, noise_lvl,
                     atmos_mode, jitter_px, plat_px):
    hud = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    cv2.drawMarker(hud, (SETPOINT_X, SETPOINT_Y), (140, 140, 140),
                   markerType=cv2.MARKER_CROSS, markerSize=24, thickness=1)
    cv2.circle(hud, (SETPOINT_X, SETPOINT_Y), int(SPEC_TRACKING_ERROR_PX), (80, 80, 80), 1)

    if cx is not None and cy is not None:
        icx, icy = int(round(cx)), int(round(cy))
        cv2.circle(hud, (icx, icy), 5, (255, 255, 0), 1)

    if kf_x is not None and kf_y is not None and state == "TRACKING":
        ikx, iky = int(round(kf_x)), int(round(kf_y))
        cv2.line(hud, (SETPOINT_X, SETPOINT_Y), (ikx, iky), (0, 165, 255), 1)
        ring_color = (0, 255, 0) if error_dist <= SPEC_TRACKING_ERROR_PX else (0, 0, 255)
        cv2.circle(hud, (ikx, iky), 8, ring_color, 2)

    status_color = (0, 255, 0) if state == "TRACKING" else (0, 255, 255)
    cv2.putText(hud, f"STATE: {state}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, status_color, 2)
    cv2.putText(hud, f"Motion: {motion_type.upper()}", (12, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 100), 1)

    cv2.putText(hud, f"Noise: {noise_mode.upper()} ({int(noise_lvl)}px) | Atmos: {atmos_mode.upper()}",
                (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1)
    cv2.putText(hud, f"Jitter: +/-{int(jitter_px)}px | Platform Drift: +/-{int(plat_px)}px",
                (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1)

    if acq_time is not None:
        spec_pass = acq_time <= SPEC_ACQ_TIME_SEC
        verdict = "PASS" if spec_pass else "FAIL"
        acq_color = (0, 255, 0) if spec_pass else (0, 0, 255)
        cv2.putText(hud, f"Acq Time: {acq_time:.2f}s (Spec <= 2s: {verdict})", (12, 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, acq_color, 1)
    else:
        current_elapsed = search_frames / FPS
        cv2.putText(hud, f"Acq Time: {current_elapsed:.2f}s (SEARCHING)", (12, 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

    cv2.putText(hud, f"Tracking Error: {error_dist:.2f} px (Spec <= 10px)", (12, 122),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
    cv2.putText(hud, f"Avg Error: {avg_error:.2f} px | Lock: {lock_rate:.1f}%", (12, 142),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
    cv2.putText(hud, f"FPS: {current_fps:.1f} (Spec >= 20) | Pan: {pan_deg:.1f} deg/s", (12, 162),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return hud


def save_performance_log(log_data, filename="performance_log.csv"):
    """
    Mandatory Deliverable 5: Auto-generates performance report containing simulation duration,
    FPS, acquisition time, average and maximum tracking error, lock retention rate, processing time.
    """
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "duration_sec", "avg_fps", "acquisition_time_sec",
            "avg_tracking_error_px", "max_tracking_error_px", "lock_retention_pct",
            "motion_type", "noise_mode", "noise_level_px", "atmosphere_mode",
            "atmosphere_severity", "jitter_px", "platform_drift_px"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)
    print(f">> Performance log saved to {filename}")


def main():
    create_control_panel()

    detector = RobustBeaconDetector()
    kf = KalmanBeaconTracker2D()
    pid = PIDController2D()

    fov_deg = 4.0
    active_motion = "horizontal"
    source = SimulatedCameraSource(target_start=(450, 1000), motion_type=active_motion)
    source.set_fov(fov_deg)

    pan_speed_deg = 8.0
    max_speed_px = compute_max_speed_px(pan_speed_deg, fov_deg)
    scanner = SpiralScanner(source.scene_width / 2, source.scene_height / 2, max_speed_px)

    state = "SEARCHING"
    frames_lost = 0
    total_tracked_frames = 0
    in_spec_frames = 0
    error_history = []
    trail_points = [(source.cam_x, source.cam_y)]
    paused = False

    search_frames = 0
    acquisition_time_sec = None

    start_time = time.time()
    total_frames = 0
    measured_fps = 30.0
    last_fps_calc = time.time()
    frames_since_calc = 0

    print("=" * 65)
    print("SIH26169 ISRO Virtual Camera Tracking Controller Active.")
    print("Interactive GUI: Use the 'Control Panel' window to tune disturbances.")
    print("Commands:")
    print("  [1]-[5] Target Motion (1=Horizontal, 2=Straight, 3=Circular, 4=Fig8, 5=Random)")
    print("  [SPACE] Pause/Resume | [R] Reset | [S] Save CSV Log | [Q] Quit")
    print("=" * 65)

    last_motion_mode = active_motion

    while True:
        (atmos_mode, atmos_sev, noise_mode, noise_lvl,
         jitter_px, plat_px, pan_deg, motion_mode) = read_control_panel()

        source.atmosphere_mode = atmos_mode
        source.atmosphere_severity = atmos_sev
        source.noise_mode = noise_mode
        source.noise_level = noise_lvl
        source.jitter_px = jitter_px
        source.platform_drift_px = plat_px

        if abs(pan_deg - pan_speed_deg) > 0.05:
            pan_speed_deg = pan_deg
            max_speed_px = compute_max_speed_px(pan_speed_deg, fov_deg)
            scanner.update_speed(max_speed_px)

        if motion_mode != last_motion_mode:
            last_motion_mode = motion_mode
            active_motion = motion_mode
            source.set_motion_type(active_motion)
            pid.reset()
            kf.reset()
            error_history.clear()
            total_tracked_frames = 0
            in_spec_frames = 0
            search_frames = 0
            acquisition_time_sec = None
            print(f">> Target Motion Switched to: {active_motion.upper()}")

        if not paused:
            frame = source.get_frame()
            if frame is None:
                break

            total_frames += 1
            frames_since_calc += 1
            if time.time() - last_fps_calc >= 1.0:
                measured_fps = frames_since_calc / (time.time() - last_fps_calc)
                frames_since_calc = 0
                last_fps_calc = time.time()

            pred_x, pred_y = kf.predict()

            cx, cy, _ = detector.detect(frame)
            kf_x, kf_y = None, None

            if cx is not None and cy is not None:
                if state == "SEARCHING":
                    if acquisition_time_sec is None:
                        acquisition_time_sec = search_frames / FPS
                    state = "TRACKING"

                frames_lost = 0
                kf_x, kf_y = kf.update(cx, cy)

                err_x = kf_x - SETPOINT_X
                err_y = kf_y - SETPOINT_Y
                current_error = math.hypot(err_x, err_y)

                error_history.append(current_error)
                total_tracked_frames += 1
                if current_error <= SPEC_TRACKING_ERROR_PX:
                    in_spec_frames += 1

                dx, dy = pid.compute(err_x, err_y, max_speed_px)
                source.pan_tilt(dx, dy)

            else:
                current_error = 0.0
                if state == "TRACKING":
                    frames_lost += 1
                    if frames_lost <= LOST_FRAME_LIMIT:
                        kf_x, kf_y = pred_x, pred_y
                        err_x = kf_x - SETPOINT_X
                        err_y = kf_y - SETPOINT_Y
                        dx, dy = pid.compute(err_x, err_y, max_speed_px)
                        source.pan_tilt(dx, dy)
                    else:
                        state = "SEARCHING"
                        pid.reset()
                        kf.reset()
                        scanner.center_x = source.cam_x
                        scanner.center_y = source.cam_y
                        scanner.r = 0.0
                        scanner.theta = 0.0

                if state == "SEARCHING":
                    search_frames += 1
                    nx, ny = scanner.next_position()
                    source.set_position(nx, ny)

            trail_points.append((source.cam_x, source.cam_y))

            avg_err = (sum(error_history) / len(error_history)) if error_history else 0.0
            lock_pct = (in_spec_frames / total_tracked_frames * 100.0) if total_tracked_frames > 0 else 0.0

            display_feed = render_dashboard(
                frame, state, active_motion, acquisition_time_sec, search_frames,
                current_error, avg_err, lock_pct, cx, cy, kf_x, kf_y,
                pan_speed_deg, fov_deg, measured_fps, source.noise_mode,
                source.noise_level, source.atmosphere_mode, source.jitter_px,
                source.platform_drift_px
            )

            overview = build_debug_view(source, display_size=500)
            scale = 500 / source.scene_width
            scaled_pts = [(int(px * scale), int(py * scale)) for px, py in trail_points[-400:]]
            for p1, p2 in zip(scaled_pts, scaled_pts[1:]):
                cv2.line(overview, p1, p2, (0, 255, 0), 1)

            cv2.imshow("Camera Viewport (640x480)", display_feed)
            cv2.imshow("World Overview", overview)

        panel_canvas = draw_control_panel_status(
            atmos_mode, atmos_sev, noise_mode, noise_lvl,
            jitter_px, plat_px, pan_deg, active_motion
        )
        cv2.imshow("Control Panel", panel_canvas)

        key = cv2.waitKey(33) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key in MOTION_NAMES:
            active_motion = MOTION_NAMES[key]
            idx = MOTION_MODES.index(active_motion)
            cv2.setTrackbarPos("Motion (0-4)", "Control Panel", idx)
            source.set_motion_type(active_motion)
            pid.reset()
            kf.reset()
            error_history.clear()
            total_tracked_frames = 0
            in_spec_frames = 0
            search_frames = 0
            acquisition_time_sec = None
            print(f">> Switched Target Motion to: {active_motion.upper()}")
        elif key in (ord('s'), ord('S')):
            duration = time.time() - start_time
            save_performance_log({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_sec": f"{duration:.2f}",
                "avg_fps": f"{measured_fps:.1f}",
                "acquisition_time_sec": f"{acquisition_time_sec:.2f}" if acquisition_time_sec else "N/A",
                "avg_tracking_error_px": f"{avg_err:.2f}",
                "max_tracking_error_px": f"{max(error_history):.2f}" if error_history else "0.00",
                "lock_retention_pct": f"{lock_pct:.1f}",
                "motion_type": active_motion,
                "noise_mode": source.noise_mode,
                "noise_level_px": f"{source.noise_level:.1f}",
                "atmosphere_mode": source.atmosphere_mode,
                "atmosphere_severity": f"{source.atmosphere_severity:.2f}",
                "jitter_px": f"{source.jitter_px:.1f}",
                "platform_drift_px": f"{source.platform_drift_px:.1f}"
            })
        elif key == ord('r'):
            source = SimulatedCameraSource(target_start=(450, 1000), motion_type=active_motion)
            source.set_fov(fov_deg)
            scanner = SpiralScanner(source.scene_width / 2, source.scene_height / 2, max_speed_px)
            pid.reset()
            kf.reset()
            state = "SEARCHING"
            frames_lost = 0
            total_tracked_frames = 0
            in_spec_frames = 0
            error_history.clear()
            search_frames = 0
            acquisition_time_sec = None
            trail_points = [(source.cam_x, source.cam_y)]

    duration = time.time() - start_time
    if total_tracked_frames > 0:
        save_performance_log({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": f"{duration:.2f}",
            "avg_fps": f"{measured_fps:.1f}",
            "acquisition_time_sec": f"{acquisition_time_sec:.2f}" if acquisition_time_sec else "N/A",
            "avg_tracking_error_px": f"{avg_err:.2f}",
            "max_tracking_error_px": f"{max(error_history):.2f}" if error_history else "0.00",
            "lock_retention_pct": f"{lock_pct:.1f}",
            "motion_type": active_motion,
            "noise_mode": source.noise_mode,
            "noise_level_px": f"{source.noise_level:.1f}",
            "atmosphere_mode": source.atmosphere_mode,
            "atmosphere_severity": f"{source.atmosphere_severity:.2f}",
            "jitter_px": f"{source.jitter_px:.1f}",
            "platform_drift_px": f"{source.platform_drift_px:.1f}"
        })

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
