import math
import time
import cv2
from step2_camera import SimulatedCameraSource, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, build_debug_view
from step4_search import SpiralScanner, target_visible

FPS = 30
VIEWPORT_HORIZONTAL = VIEWPORT_WIDTH

PRESETS = {
    0: {"name": "Default (far-left)", "target_start": (200, 1000)},
    1: {"name": "Edge Case (blue band)", "target_start": (640, 708)},
}

def create_control_window():
    cv2.namedWindow("Controls")
    cv2.createTrackbar("Pan Speed x10 (5.0-10.0 deg/s)", "Controls", 80, 100, lambda x: None)
    cv2.createTrackbar("FOV x10 (2.0-8.0 deg)", "Controls", 40, 80, lambda x: None)
    cv2.createTrackbar("Preset (0=Default,1=EdgeCase)", "Controls", 0, 1, lambda x: None)

def read_preset():
    return cv2.getTrackbarPos("Preset (0=Default,1=EdgeCase)", "Controls")

def read_controls():
    pan_speed_deg = max(cv2.getTrackbarPos("Pan Speed x10 (5.0-10.0 deg/s)", "Controls"), 50) / 10.0
    fov_deg = max(cv2.getTrackbarPos("FOV x10 (2.0-8.0 deg)", "Controls"), 20) / 10.0
    return pan_speed_deg, fov_deg

def compute_max_speed_px(pan_speed_deg, fov_deg, fps=FPS):
    px_per_degree = VIEWPORT_HORIZONTAL / fov_deg
    return (pan_speed_deg / fps) * px_per_degree

def draw_dashboard(frame, phase, elapsed_sec, frame_count, pan_speed_deg, fov_deg):
    display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    color = (0, 255, 255) if phase == "SEARCHING" else (0, 255, 0)
    cv2.putText(display, f"Phase: {phase}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(display, f"Elapsed: {elapsed_sec:.2f}s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(display, f"Frame: {frame_count}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(display, f"Pan/Tilt: {pan_speed_deg:.1f} deg/s  FOV: {fov_deg:.1f} deg",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if phase == "ACQUIRED":
        spec_result = "PASS" if elapsed_sec <= 2.0 else "FAIL"
        spec_color = (0, 255, 0) if spec_result == "PASS" else (0, 0, 255)
        cv2.putText(display, f"Spec <=2s: {spec_result}", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, spec_color, 2)
    return display

def draw_debug_with_trail(source, trail_points, display_size=500):
    debug = build_debug_view(source, display_size)
    scale = display_size / source.scene_width
    scaled_points = [(int(x * scale), int(y * scale)) for x, y in trail_points]
    for p1, p2 in zip(scaled_points, scaled_points[1:]):
        cv2.line(debug, p1, p2, (0, 200, 0), 1)
    return debug

def build_search(preset_idx, pan_speed_deg, fov_deg):
    preset = PRESETS[preset_idx]
    source = SimulatedCameraSource(target_start=preset["target_start"], motion_type="horizontal")
    speed_px = compute_max_speed_px(pan_speed_deg, fov_deg)
    scanner = SpiralScanner(source.scene_width / 2, source.scene_height / 2, speed_px)
    return source, scanner

def main():
    pan_speed_deg, fov_deg = 8.0, 4.0
    create_control_window()
    current_preset = read_preset()
    source, scanner = build_search(current_preset, pan_speed_deg, fov_deg)

    phase = "SEARCHING"
    frame_count = 0
    trail_points = [(source.cam_x, source.cam_y)]
    paused = False
    last_frame = source.get_frame()
    acquired_at = None

    while True:
        pan_speed_deg, fov_deg = read_controls()
        selected_preset = read_preset()
        source.set_fov(fov_deg)

        if selected_preset != current_preset:
            current_preset = selected_preset
            source, scanner = build_search(current_preset, pan_speed_deg, fov_deg)
            phase = "SEARCHING"
            frame_count = 0
            trail_points = [(source.cam_x, source.cam_y)]
            last_frame = source.get_frame()
            acquired_at = None

        if not paused:
            scanner.update_speed(compute_max_speed_px(pan_speed_deg, fov_deg))
            frame = source.get_frame()
            last_frame = frame

            if phase == "SEARCHING" and target_visible(frame):
                phase = "ACQUIRED"
                acquired_at = frame_count

            if phase == "SEARCHING":
                x, y = scanner.next_position()
                source.set_position(x, y)
                trail_points.append((source.cam_x, source.cam_y))

            frame_count += 1

        elapsed_sec = frame_count / FPS
        display_phase = "PAUSED" if paused else phase
        display = draw_dashboard(last_frame, display_phase, elapsed_sec,
                                  frame_count, pan_speed_deg, fov_deg)
        cv2.imshow("Camera Feed - Acquisition Dashboard", display)
        cv2.imshow("Full World View (with search trail)",
                   draw_debug_with_trail(source, trail_points))

        key = cv2.waitKey(33) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('r'):
            source, scanner = build_search(current_preset, pan_speed_deg, fov_deg)
            phase = "SEARCHING"
            frame_count = 0
            trail_points = [(source.cam_x, source.cam_y)]
            last_frame = source.get_frame()
            paused = False
            acquired_at = None

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
