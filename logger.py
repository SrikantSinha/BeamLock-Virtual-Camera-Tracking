"""
logger.py — Performance and Telemetry Logging System for SIH26169
Mandatory Deliverable 5: Generates streaming CSV frame telemetry and
summary JSON performance reports with benchmark validation.
"""

import os
import csv
import json
import time
import math
import numpy as np


class PerformanceLogger:
    def __init__(self, output_dir="logs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        session_id = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.output_dir, f"telemetry_{session_id}.csv")
        self.json_path = os.path.join(self.output_dir, f"report_{session_id}.json")

        self.csv_file = open(self.csv_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "timestamp", "frame_id", "state", "motion_type",
            "target_x", "target_y", "cam_x", "cam_y",
            "pan_deg", "tilt_deg", "error_px", "kalman_error_px",
            "fps", "latency_ms", "noise_mode", "atmosphere_mode"
        ])

        # Session tracking metrics
        self.start_time = time.time()
        self.frame_count = 0
        self.tracking_frames = 0
        self.in_spec_frames = 0
        self.acquisition_time = None
        self.reacquisition_times = []
        self.tracking_errors = []
        self.latencies = []

    def log_frame(self, frame_id, state, motion_type, target_pos, cam_pos,
                  pan_deg, tilt_deg, error_px, kalman_error_px, fps, latency_ms,
                  noise_mode, atmosphere_mode):
        """Streams single-frame telemetry directly to CSV."""
        self.frame_count += 1
        self.latencies.append(latency_ms)

        if state == "TRACKING":
            self.tracking_frames += 1
            if error_px is not None:
                self.tracking_errors.append(error_px)
                if error_px <= 10.0:
                    self.in_spec_frames += 1

        self.csv_writer.writerow([
            f"{time.time() - self.start_time:.3f}",
            frame_id, state, motion_type,
            f"{target_pos[0]:.2f}", f"{target_pos[1]:.2f}",
            f"{cam_pos[0]:.2f}", f"{cam_pos[1]:.2f}",
            f"{pan_deg:.2f}", f"{tilt_deg:.2f}",
            f"{error_px:.2f}" if error_px is not None else "NaN",
            f"{kalman_error_px:.2f}" if kalman_error_px is not None else "NaN",
            f"{fps:.1f}", f"{latency_ms:.2f}",
            noise_mode, atmosphere_mode
        ])

    def record_acquisition(self, elapsed_sec):
        if self.acquisition_time is None:
            self.acquisition_time = elapsed_sec

    def record_reacquisition(self, elapsed_sec):
        self.reacquisition_times.append(elapsed_sec)

    def generate_summary(self):
        """Compiles session statistics into an official benchmark JSON report."""
        duration = time.time() - self.start_time
        errors = np.array(self.tracking_errors) if self.tracking_errors else np.array([0.0])
        latencies = np.array(self.latencies) if self.latencies else np.array([0.0])

        rmse = float(np.sqrt(np.mean(errors**2))) if len(errors) > 0 else 0.0
        avg_err = float(np.mean(errors)) if len(errors) > 0 else 0.0
        max_err = float(np.max(errors)) if len(errors) > 0 else 0.0
        lock_rate = (self.in_spec_frames / self.tracking_frames * 100.0) if self.tracking_frames > 0 else 0.0
        target_loss_rate = 100.0 - lock_rate

        report = {
            "metadata": {
                "system": "AI-Based Virtual Camera Tracking System (SIH26169)",
                "organization": "ISRO / Department of Space",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": round(duration, 2),
                "total_frames_processed": self.frame_count
            },
            "performance_benchmarks": {
                "acquisition_time_sec": round(self.acquisition_time, 2) if self.acquisition_time else "N/A",
                "acquisition_spec_passed": bool(self.acquisition_time <= 2.0) if self.acquisition_time else False,
                "reacquisition_time_avg_sec": round(float(np.mean(self.reacquisition_times)), 2) if self.reacquisition_times else 0.0,
                "reacquisition_spec_passed": bool(np.mean(self.reacquisition_times) <= 1.0) if self.reacquisition_times else True,
                "rmse_tracking_error_px": round(rmse, 2),
                "average_tracking_error_px": round(avg_err, 2),
                "maximum_tracking_error_px": round(max_err, 2),
                "tracking_error_spec_passed": bool(rmse <= 10.0),
                "target_loss_rate_percent": round(target_loss_rate, 2),
                "target_loss_spec_passed": bool(target_loss_rate < 5.0),
                "lock_retention_rate_percent": round(lock_rate, 2),
                "average_fps": round(float(self.frame_count / duration), 1) if duration > 0 else 0.0,
                "fps_spec_passed": bool((self.frame_count / duration) >= 20.0) if duration > 0 else False,
                "average_processing_latency_ms": round(float(np.mean(latencies)), 2)
            }
        }

        with open(self.json_path, "w") as f:
            json.dump(report, f, indent=4)

        self.csv_file.close()
        return report, self.csv_path, self.json_path