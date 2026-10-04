import math
import cv2
from step2_camera import SimulatedCameraSource, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, build_debug_view
from step4_search import SpiralScanner, target_visible

FPS = 30
FOV_DEG = 4
PX_PER_DEGREE = VIEWPORT_WIDTH / FOV_DEG
MAX_PAN_SPEED_DEG = 8
MAX_SPEED_PX_PER_FRAME = (MAX_PAN_SPEED_DEG / FPS) * PX_PER_DEGREE

TARGET_START = (640, 708)

def run_edge_test():
    source = SimulatedCameraSource(target_start=TARGET_START, motion_type="horizontal")
    scanner = SpiralScanner(source.scene_width / 2, source.scene_height / 2, MAX_SPEED_PX_PER_FRAME)

    trail = [(source.cam_x, source.cam_y)]
    for frame_count in range(300):
        frame = source.get_frame()
        if target_visible(frame):
            seconds = frame_count / FPS
            print(f"Target acquired in {frame_count} frames (~{seconds:.2f}s)")
            print(f"Spec <=2s: {'PASS' if seconds <= 2 else 'FAIL'}")
            return source, trail, frame_count

        x, y = scanner.next_position()
        source.set_position(x, y)
        trail.append((source.cam_x, source.cam_y))

    print("Not found within 10 seconds")
    return source, trail, None

if __name__ == "__main__":
    source, trail, frames = run_edge_test()
