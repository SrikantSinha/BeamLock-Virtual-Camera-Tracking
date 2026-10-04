"""Core CV components from the original dashboard; no GUI toolkit."""

import math
import cv2
import numpy as np
from step2_camera import VIEWPORT_WIDTH, VIEWPORT_HEIGHT

# ---- Hardware & Display Constants ----
PIXELS_PER_DEGREE = 160.0        # 640px / 4.0 deg FOV
SETPOINT_X = VIEWPORT_WIDTH // 2 # 320 px
SETPOINT_Y = VIEWPORT_HEIGHT // 2# 240 px


class RobustBeaconDetector:
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
            return candidates[0][1], candidates[0][2], candidates[0][0]

        return None, None, 0


class KalmanBeaconTracker2D:
    def __init__(self, dt=1.0 / 30.0):
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
        if cmd_magnitude > max_speed_px and cmd_magnitude > 0:
            scale = max_speed_px / cmd_magnitude
            cmd_x *= scale
            cmd_y *= scale
        else:
            self.integral_x += error_x
            self.integral_y += error_y

        return cmd_x, cmd_y


