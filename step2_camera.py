"""
step2_camera.py — Virtual Optical Scene, Multi-Pattern Target, Movable Viewport,
and ISRO-Specified Disturbances & Noise Engine.
Specifications (ISRO PDF Sr. No. 1-15, 21-25):
  - Screen Size: 2000x2000 px, Monochrome
  - Viewport Resolution: 640x480 px, Monochrome
  - Target Motion: Horizontal, Straight Line, Circular, Figure-of-8, Random
  - Image Noise: Salt & Pepper (~10%), Gaussian (std <= 20), Poisson
  - Camera Jitter: +/-20 px/frame high-frequency vibration
  - Atmospheric Disturbance: Clear, Haze, Fog, Rain, Low Light
  - Platform Motion: +/-20 px/frame carrier drift
"""

import math
import cv2
import numpy as np

VIEWPORT_WIDTH = 640
VIEWPORT_HEIGHT = 480
FOV_REFERENCE_DEG = 4.0  # 640px = 4 deg (160 px/degree)


class SimulatedCameraSource:
    def __init__(self, scene_width=2000, scene_height=2000,
                 target_start=None, motion_type="horizontal"):
        self.scene_width = scene_width
        self.scene_height = scene_height
        self.target_size = 10  # Spec: 5-20 px, default 10x10

        if target_start is not None:
            self.target_x, self.target_y = float(target_start[0]), float(target_start[1])
        else:
            self.target_x, self.target_y = 450.0, float(scene_height // 2)

        # Motion models
        # 1. Pure Horizontal Straight Path
        self.horiz_vx = 3.2

        # 2. Straight Line (Diagonal Bounce)
        self.linear_vx = 3.0
        self.linear_vy = 1.8

        # 3. Circular Orbit
        self.circle_center_x = scene_width / 2.0
        self.circle_center_y = scene_height / 2.0
        self.circle_radius = 500.0
        self.circle_omega = 0.0065
        self.circle_theta = 0.0

        # 4. Figure-of-8 (Lemniscate)
        self.fig8_center_x = scene_width / 2.0
        self.fig8_center_y = scene_height / 2.0
        self.fig8_a = 520.0
        self.fig8_b = 280.0
        self.fig8_omega = 0.007
        self.fig8_theta = 0.0

        # 5. Inertial Random Walk
        self.rand_vx = 2.0
        self.rand_vy = 1.5
        self.max_rand_speed = 3.2

        # Viewport parameters
        self.fov_deg = FOV_REFERENCE_DEG
        self.cam_x = scene_width // 2
        self.cam_y = scene_height // 2

        # Disturbances and Noise configuration
        self.noise_mode = "none"           # "none", "salt_pepper", "gaussian", "poisson"
        self.noise_level = 10.0            # 0 to 20 px std (Gaussian) or % (S&P)
        self.atmosphere_mode = "clear"     # "clear", "haze", "fog", "rain", "low_light"
        self.atmosphere_severity = 0.75    # 0.0 to 1.0 scaling factor
        self.jitter_px = 0.0               # 0.0 to 20.0 px/frame
        self.platform_drift_px = 0.0       # 0.0 to 20.0 px/frame
        self.platform_t = 0.0

        self.set_motion_type(motion_type)

    def set_motion_type(self, motion_type):
        self.motion_type = motion_type
        if motion_type == "horizontal":
            if abs(self.horiz_vx) < 1.0:
                self.horiz_vx = 3.2
        elif motion_type == "circular":
            dist = math.hypot(self.target_x - self.circle_center_x,
                              self.target_y - self.circle_center_y)
            if dist > 50:
                self.circle_radius = dist
                self.circle_theta = math.atan2(self.target_y - self.circle_center_y,
                                               self.target_x - self.circle_center_x)
            else:
                self.circle_radius = 450.0
                self.circle_theta = 0.0
                self.target_x = self.circle_center_x + self.circle_radius
                self.target_y = self.circle_center_y
        elif motion_type == "figure_8":
            self.fig8_theta = 0.0
            self.target_x = self.fig8_center_x
            self.target_y = self.fig8_center_y
        elif motion_type == "straight_line":
            if abs(self.linear_vx) < 1.0:
                self.linear_vx = 3.0
            if abs(self.linear_vy) < 1.0:
                self.linear_vy = 1.8

    def _update_target_position(self):
        margin = 60
        if self.motion_type == "horizontal":
            self.target_x += self.horiz_vx
            if self.target_x > self.scene_width - margin:
                self.target_x = self.scene_width - margin
                self.horiz_vx = -abs(self.horiz_vx)
            elif self.target_x < margin:
                self.target_x = margin
                self.horiz_vx = abs(self.horiz_vx)

        elif self.motion_type == "straight_line":
            self.target_x += self.linear_vx
            self.target_y += self.linear_vy
            if self.target_x > self.scene_width - margin or self.target_x < margin:
                self.linear_vx = -self.linear_vx
            if self.target_y > self.scene_height - margin or self.target_y < margin:
                self.linear_vy = -self.linear_vy

        elif self.motion_type == "circular":
            self.circle_theta += self.circle_omega
            self.target_x = self.circle_center_x + self.circle_radius * math.cos(self.circle_theta)
            self.target_y = self.circle_center_y + self.circle_radius * math.sin(self.circle_theta)

        elif self.motion_type == "figure_8":
            self.fig8_theta += self.fig8_omega
            self.target_x = self.fig8_center_x + self.fig8_a * math.sin(self.fig8_theta)
            self.target_y = self.fig8_center_y + self.fig8_b * math.sin(2.0 * self.fig8_theta)

        elif self.motion_type == "random":
            self.rand_vx += float(np.random.uniform(-0.3, 0.3))
            self.rand_vy += float(np.random.uniform(-0.3, 0.3))
            speed = math.hypot(self.rand_vx, self.rand_vy)
            if speed > self.max_rand_speed:
                self.rand_vx = (self.rand_vx / speed) * self.max_rand_speed
                self.rand_vy = (self.rand_vy / speed) * self.max_rand_speed

            if self.target_x < margin:
                self.rand_vx = abs(self.rand_vx)
            elif self.target_x > self.scene_width - margin:
                self.rand_vx = -abs(self.rand_vx)
            if self.target_y < margin:
                self.rand_vy = abs(self.rand_vy)
            elif self.target_y > self.scene_height - margin:
                self.rand_vy = -abs(self.rand_vy)

            self.target_x += self.rand_vx
            self.target_y += self.rand_vy

    def _build_full_scene(self):
        scene = np.zeros((self.scene_height, self.scene_width), dtype=np.uint8)
        half = self.target_size // 2
        x1 = int(round(self.target_x - half))
        y1 = int(round(self.target_y - half))
        x2 = int(round(self.target_x + half))
        y2 = int(round(self.target_y + half))
        cv2.rectangle(scene, (x1, y1), (x2, y2), 255, -1)
        return scene

    def pan_tilt(self, dx, dy):
        self.cam_x = int(np.clip(self.cam_x + dx, 0, self.scene_width))
        self.cam_y = int(np.clip(self.cam_y + dy, 0, self.scene_height))

    def set_position(self, x, y):
        self.cam_x = int(np.clip(x, 0, self.scene_width))
        self.cam_y = int(np.clip(y, 0, self.scene_height))

    def set_fov(self, fov_deg):
        self.fov_deg = max(1.0, fov_deg)

    def get_frame(self):
        self._update_target_position()
        full_scene = self._build_full_scene()

        # Dynamic Jitter perturbation: +/- jitter_px per frame
        jit_x, jit_y = 0, 0
        if self.jitter_px > 0:
            jit_x = int(round(np.random.uniform(-self.jitter_px, self.jitter_px)))
            jit_y = int(round(np.random.uniform(-self.jitter_px, self.jitter_px)))

        # Dynamic Platform carrier drift: sinusoidal linear sweep up to +/- platform_drift_px
        plat_x, plat_y = 0, 0
        if self.platform_drift_px > 0:
            self.platform_t += 0.05
            plat_x = int(round(self.platform_drift_px * math.sin(self.platform_t)))
            plat_y = int(round(self.platform_drift_px * 0.65 * math.cos(0.7 * self.platform_t)))

        eff_cam_x = self.cam_x + jit_x + plat_x
        eff_cam_y = self.cam_y + jit_y + plat_y

        zoom_factor = self.fov_deg / FOV_REFERENCE_DEG
        crop_w = int(VIEWPORT_WIDTH * zoom_factor)
        crop_h = int(VIEWPORT_HEIGHT * zoom_factor)

        x1 = int(round(eff_cam_x - crop_w / 2.0))
        y1 = int(round(eff_cam_y - crop_h / 2.0))
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - self.scene_width)
        pad_bottom = max(0, y2 - self.scene_height)

        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(self.scene_width, x2)
        src_y2 = min(self.scene_height, y2)

        cropped = full_scene[src_y1:src_y2, src_x1:src_x2]

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            raw_crop = cv2.copyMakeBorder(
                cropped, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=0
            )
        else:
            raw_crop = cropped

        frame = cv2.resize(raw_crop, (VIEWPORT_WIDTH, VIEWPORT_HEIGHT))

        # Atmospheric disturbance pipeline (Sr. No. 24) scaled by severity
        if self.atmosphere_mode != "clear":
            img = frame.astype(np.float32)
            sev = float(np.clip(self.atmosphere_severity, 0.0, 1.0))
            if self.atmosphere_mode == "haze":
                alpha = 1.0 - 0.45 * sev
                beta = 55.0 * sev
                img = img * alpha + beta
            elif self.atmosphere_mode == "fog":
                alpha = 1.0 - 0.70 * sev
                beta = 100.0 * sev
                img = img * alpha + beta
            elif self.atmosphere_mode == "low_light":
                alpha = max(0.20, 1.0 - 0.75 * sev)
                img = img * alpha
            elif self.atmosphere_mode == "rain":
                alpha = 1.0 - 0.35 * sev
                img = img * alpha + 20.0 * sev
                num_drops = int(300 * sev)
                for _ in range(num_drops):
                    rx = np.random.randint(0, frame.shape[1] - 10)
                    ry = np.random.randint(0, frame.shape[0] - 20)
                    cv2.line(img, (rx, ry), (rx + 4, ry + 16), 180.0, 1)
            frame = np.clip(img, 0, 255).astype(np.uint8)

        # Image noise injection pipeline (Sr. No. 21, 22) scaled by noise_level
        lvl = float(self.noise_level)
        if self.noise_mode == "gaussian" and lvl > 0:
            noise = np.random.normal(0, lvl, frame.shape).astype(np.float32)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        elif self.noise_mode == "salt_pepper" and lvl > 0:
            prob = lvl / 100.0
            rnd = np.random.rand(*frame.shape)
            frame = frame.copy()
            frame[rnd < (prob / 2.0)] = 0
            frame[rnd > (1.0 - prob / 2.0)] = 255
        elif self.noise_mode == "poisson" and lvl > 0:
            vals = max(int(255.0 / max(lvl, 1.0) * 4.0), 2)
            vals = 2 ** np.ceil(np.log2(vals))
            frame = np.clip(np.random.poisson(frame.astype(np.float32) / 255.0 * vals) / float(vals) * 255.0, 0, 255).astype(np.uint8)

        return frame


class VideoFileSource:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def build_debug_view(source, display_size=500):
    full_scene = source._build_full_scene()
    debug = cv2.cvtColor(full_scene, cv2.COLOR_GRAY2BGR)

    zoom_factor = source.fov_deg / FOV_REFERENCE_DEG
    crop_w = int(VIEWPORT_WIDTH * zoom_factor)
    crop_h = int(VIEWPORT_HEIGHT * zoom_factor)

    x1 = int(round(source.cam_x - crop_w / 2.0))
    y1 = int(round(source.cam_y - crop_h / 2.0))
    cv2.rectangle(debug, (x1, y1), (x1 + crop_w, y1 + crop_h),
                  (0, 0, 255), thickness=4)

    scale = display_size / source.scene_width
    debug = cv2.resize(debug, (display_size, int(source.scene_height * scale)))
    return debug
