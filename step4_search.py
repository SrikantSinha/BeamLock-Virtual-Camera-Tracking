import math
import cv2
from step2_camera import SimulatedCameraSource, VIEWPORT_WIDTH, VIEWPORT_HEIGHT


class SpiralScanner:
    """
    FOV-optimized Archimedean Spiral Search Generator.
    Tailors the radial growth pitch to the camera viewport footprint (640x480)
    to eliminate redundant scanning and maximize area coverage per second,
    strictly adhering to physical motor velocity constraints (deg/s -> px/frame).
    """

    def __init__(self, center_x, center_y, max_speed_px_per_frame,
                 half_width=None, half_height=None, fov_w=640, fov_h=480, overlap=0.30):
        self.center_x = center_x
        self.center_y = center_y
        self.max_speed = max_speed_px_per_frame
        
        # Spiral pitch corresponds to optical viewport height minus overlap
        self.pitch = fov_h * (1.0 - overlap) # ~336 px per revolution
        self.b = self.pitch / (2.0 * math.pi) # ~53.47 px/rad
        
        self.theta = 0.0
        self.r = 0.0

    def update_speed(self, max_speed_px_per_frame):
        """Allows dynamic pan speed adjustment without resetting search progress."""
        self.max_speed = max_speed_px_per_frame

    def next_position(self):
        """
        Computes next camera setpoint maintaining constant linear speed (v_max)
        along the continuous Archimedean spiral trajectory.
        """
        denom = math.sqrt(self.b**2 + self.r**2)
        d_theta = self.max_speed / max(denom, 1.0)
        
        self.theta += d_theta
        self.r = self.b * self.theta
        
        x = self.center_x + self.r * math.cos(self.theta)
        y = self.center_y + self.r * math.sin(self.theta)
        return x, y


def target_visible(frame, brightness_threshold=200):
    """Fast binary threshold check for optical presence."""
    _, thresh = cv2.threshold(frame, brightness_threshold, 255, cv2.THRESH_BINARY)
    return cv2.countNonZero(thresh) > 0


def search_for_target(source, fps=30, max_seconds=10,
                       max_pan_speed_deg=8.0, fov_deg=4.0):
    """
    Executes spiral scan until optical beacon enters camera field of view.
    Returns (frames_elapsed, cam_x, cam_y).
    """
    px_per_degree = VIEWPORT_WIDTH / fov_deg
    max_speed_px = (max_pan_speed_deg / fps) * px_per_degree

    scanner = SpiralScanner(source.scene_width / 2, source.scene_height / 2, max_speed_px)
    max_frames = int(fps * max_seconds)

    for frame_count in range(max_frames):
        frame = source.get_frame()
        if target_visible(frame):
            return frame_count, source.cam_x, source.cam_y

        x, y = scanner.next_position()
        source.set_position(x, y)

    return None, None, None
