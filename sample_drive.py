import socket
import threading
import struct
import cv2
import numpy as np
import time
import keyboard
import select
import ctypes
import random

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CAMERA_HOST = '127.0.0.1'
FRONT_CAMERA_PORT = 8080
BACK_CAMERA_PORT = 8082
CONTROL_HOST = '127.0.0.1'
CONTROL_PORT = 8081
START_LANE = 0
START_CENTER_HOLD_SECONDS = 2.0
CAR_ACCELERATION = 0.96
GREEN_CHASE_ACCELERATION = 0.70
GREEN_STEER_GAIN = 4.0
GREEN_STEER_DEADZONE = 0.015
GREEN_MIN_STEER = 0.18
MIN_LOOKAHEAD_Y_RATIO = 0.25
LINE_GROUP_Y_RATIO = 0.08
LANE_CHANGE_TIME_Y_RATIO = 0.14
GREEN_LOCK_SECONDS = 1.6
STEER_TAP_LOOPS = 12
STEER_RESET_LOOPS = 4
TOKEN_DECISION_Y_MIN = 25
LANE_CHANGE_TAP_LOOPS = 35
LANE_CHANGE_RESET_LOOPS = 6
TOKEN_TARGET_MEMORY_SECONDS = 2.4
GREEN_TARGET_LOCK_SECONDS = 1.1
YELLOW_EFFECT_DURATION_SECONDS = 5.0
TOKEN_COLLECTION_Y_RATIO = 0.90
TOKEN_COLLECTION_COOLDOWN_SECONDS = 2.0
HAZARD_AVOID_Y_MIN = 25
MIN_ACCELERATION_WHEN_SLOWED = 0.76
MIN_STEERING_ACCELERATION = 0.56
GREEN_APPROACH_STEER_DEADZONE = 0.04
GREEN_APPROACH_MIN_STEER = 0.65
GREEN_APPROACH_MAX_ACCELERATION = 0.58
TOKEN_VISIBLE_MAX_ACCELERATION = 0.62
HIGH_SPEED_MAX_ACCELERATION = 0.72
HAZARD_AVOID_STEER = 1.0
HAZARD_AVOID_MAX_ACCELERATION = 0.55
GREEN_TOKEN_MIN_Y_RATIO = 0.34
TOKEN_PROCESS_SCALE = 1.0
ROAD_PROCESS_SCALE = 0.55
ROAD_DETECTION_STRIDE = 4
ROAD_SKIP_IF_SLOW_MS = 45.0
TOKEN_ROW_GROUP_Y_RATIO = 0.12
HAZARD_EMERGENCY_Y_RATIO = 0.40
HAZARD_SIDE_BUFFER_Y_RATIO = 0.08
DECISION_LOCK_SECONDS = 0.25
HAZARD_DECISION_LOCK_SECONDS = 0.42
HAZARD_DIRECT_CENTER_RATIO = 0.26
HAZARD_DIRECT_Y_RATIO = 0.34
GREEN_HAZARD_BLOCK_X_RATIO = 0.18
GREEN_HAZARD_BLOCK_Y_MARGIN_RATIO = 0.10
HAZARD_LOCK_ACCELERATION = 0.50
CAMERA_DELAY_SECONDS = 5.0
ACTION_DELAY_SECONDS = 5.0
YELLOW_EFFECTS = [
    'hide_next_token_type',
    'tokens_invisible',
    'camera_input_delay',
    'action_output_delay',
    'corrupted_camera_input'
]

# Shared Resources with Mutex Lock for Concurrency
shared_data = {
    'latest_front_frame': None,
    'latest_back_frame': None,
    'steering_input' : 0.0,
    'acceleration_input' : 0.0,
    'detected_tokens': [],
    'detected_green_road_lines': [],
    'road_center_x': None,
    'processing_ms': 0.0,
    'processing_fps': 0.0,
    'selected_green_target': None,
    'selected_green_target_type': 'NONE',
    'navigation_target_x': None,
    'navigation_target_y': None,
    'navigation_mode': 'lane following',
    'slow_reason': 'none',
    'target_lane': START_LANE, # Default: Stay in Center Lane (0)
    'danger_detected': False,  # Default: No trailing car danger
    'decision_debug': '',
    'green_token_count': 0,
    'last_green_detected_time': 0.0,
    'green_target_source': 'none',
    'hazard_debug': 'none'
}

# Additional shared flags / history for trailing/police detection
shared_data.update({
    'police_detected': False,
    'trailing_detected': False,
    'back_prev_gray': None,
    'back_prev_area': 0.0,
    'run_summary': {
        'green_collected': 0,
        'red_collected': 0,
        'yellow_collected': 0,
        'yellow_effects': {
            'hide_next_token_type': 0,
            'tokens_invisible': 0,
            'camera_input_delay': 0,
            'action_output_delay': 0,
            'corrupted_camera_input': 0
        },
        'police_appeared': False,
        'trailing_appeared': False
    },
    'active_yellow_effect': None,
    'yellow_effect_until': 0.0,
    'hidden_next_token_type': False,
    'last_collected_time': 0.0,
    'last_collected_line_id': None,
    'last_collected_by_lane_color': {}
})
data_lock = threading.Lock()
is_running = True
front_camera_delay_buffer = []
action_delay_buffer = []

# Back-camera detection tuning
TRAILING_MOTION_AREA_MIN = 3500
TRAILING_APPROACH_RATIO = 1.15
POLICE_MIN_COLOR_AREA = 300
POLICE_ROI_X_START = 0.20
POLICE_ROI_X_END = 0.80
POLICE_ROI_Y_START = 0.35
POLICE_ROI_Y_END = 0.95

# ---------------------------------------------------------
# Real-Time Scheduling Framework (Do not change this in your code)
# ---------------------------------------------------------
class TaskPriority:
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class RTTask(threading.Thread):
    """
    Real-Time Task implementing:
    - Concurrency (inherits threading.Thread)
    - Task Period (enforced in run loop)
    - Task Priority (logical priority assigned)
    """
    def __init__(self, name, period, priority, execute_func):
        super().__init__()
        self.name = name
        self.period = period
        self.priority = priority
        self.execute_func = execute_func
        self.daemon = True

    def run(self):
        print(f"[{self.name}] Started | Period: {self.period}s | Priority: {self.priority}")
        try:
            handle = ctypes.windll.kernel32.GetCurrentThread()
            if self.priority == TaskPriority.HIGH:
                ctypes.windll.kernel32.SetThreadPriority(handle, 2)
            elif self.priority == TaskPriority.MEDIUM:
                ctypes.windll.kernel32.SetThreadPriority(handle, 0)
            elif self.priority == TaskPriority.LOW:
                ctypes.windll.kernel32.SetThreadPriority(handle, -2)
        except Exception as e:
            pass

        while is_running:
            start_time = time.time()
            self.execute_func()
            exec_time = time.time() - start_time
            sleep_time = self.period - exec_time
            
            if sleep_time > 0:
                time.sleep(sleep_time)

# ---------------------------------------------------------
# Network Connection Setup (Do not change this in your code)
# ---------------------------------------------------------
front_camera_sock = None
back_camera_sock = None
control_conn = None

def setup_cameras():
    global front_camera_sock, back_camera_sock
    
    print("Connecting to Cameras...")
    front_connected = False
    back_connected = False
    
    while is_running and not (front_connected and back_connected):
        if not front_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, FRONT_CAMERA_PORT))
                front_camera_sock = s
                print("Connected to Front Camera successfully.")
                front_connected = True
            except Exception:
                pass
                
        if not back_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, BACK_CAMERA_PORT))
                back_camera_sock = s
                print("Connected to Back Camera successfully.")
                back_connected = True
            except Exception:
                pass
                
        if not (front_connected and back_connected):
            time.sleep(1)

def setup_control_server():
    global control_conn
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((CONTROL_HOST, CONTROL_PORT))
    server_sock.listen()
    server_sock.settimeout(1.0)
    print(f"Control server listening on {CONTROL_HOST}:{CONTROL_PORT}")
    
    while is_running:
        try:
            conn, addr = server_sock.accept()
            print(f"Control client connected from {addr}")
            control_conn = conn
            break
        except socket.timeout:
            continue

# ---------------------------------------------------------
# Task Implementations (This is where you write your tasks)
# ---------------------------------------------------------

def read_single_camera(sock, window_name, data_key, display=True):
    #This function reads the latest frame from the camera socket and stores it in the shared data
    if sock is None:
        return
        
    try:
        latest_frame_data = None
        sock.settimeout(None)
        length_bytes = sock.recv(4)
        if not length_bytes:
            return
            
        image_length = int.from_bytes(length_bytes, 'little')
        received_bytes = b''
        while len(received_bytes) < image_length and is_running:
            packet = sock.recv(image_length - len(received_bytes))
            if not packet:
                break
            received_bytes += packet
            
        if len(received_bytes) == image_length:
            latest_frame_data = received_bytes
            
        while is_running:
            readable, _, _ = select.select([sock], [], [], 0.0)
            if not readable:
                break
                
            sock.settimeout(1.0)
            length_bytes = sock.recv(4)
            if not length_bytes:
                return
            image_length = int.from_bytes(length_bytes, 'little')
            received_bytes = b''
            while len(received_bytes) < image_length and is_running:
                packet = sock.recv(image_length - len(received_bytes))
                if not packet:
                    break
                received_bytes += packet
                
            if len(received_bytes) == image_length:
                latest_frame_data = received_bytes
                
        if latest_frame_data is not None:
            np_arr = np.frombuffer(latest_frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                with data_lock:
                    shared_data[data_key] = frame
                
                if display:
                    # You may disable this if you don't need to display the frames / This could effect the fps
                    frame_resized = cv2.resize(frame, (640, 480))
                    cv2.imshow(window_name, frame_resized)
                    cv2.waitKey(1)
                
    except Exception as e:
        pass

def read_front_camera_task():
    read_single_camera(front_camera_sock, "Front Camera", 'latest_front_frame', display=False)

def read_back_camera_task():
    read_single_camera(back_camera_sock, "Back Camera", 'latest_back_frame', display=False)
def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i/255.0) ** invGamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def build_token_exclusion_mask(tokens, frame_shape):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    for token in tokens:
        radius = int(token.get("radius", 0))
        if radius <= 0:
            continue
        center = (int(token.get("x", 0)), int(token.get("y", 0)))
        cv2.circle(mask, center, radius + 6, 255, -1)
    return mask

def detect_road_lines(frame, token_exclusion_mask=None):
    frame_h, frame_w = frame.shape[:2]
    roi_start = int(frame_h * 0.38)
    roi = frame[roi_start:, :]

    roi_bright = cv2.GaussianBlur(adjust_gamma(roi, gamma=1.15), (5, 5), 0)
    hsv_roi = cv2.cvtColor(roi_bright, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(
        hsv_roi,
        np.array((36, 32, 90), dtype=np.uint8),
        np.array((90, 255, 255), dtype=np.uint8)
    )

    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    green_mask = cv2.dilate(green_mask, np.ones((5, 5), np.uint8), iterations=1)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, np.ones((13, 3), np.uint8))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, np.ones((3, 13), np.uint8))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    road_mask = green_mask

    if token_exclusion_mask is not None:
        token_roi = token_exclusion_mask[roi_start:, :]
        road_mask = cv2.bitwise_and(road_mask, cv2.bitwise_not(token_roi))

    edges = cv2.Canny(road_mask, 40, 120)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=26,
        minLineLength=max(34, int(frame_w * 0.08)),
        maxLineGap=max(14, int(frame_w * 0.04))
    )

    road_lines = []
    if lines is None:
        return road_lines, road_mask, roi_start, None

    min_length = max(45.0, frame_w * 0.10)
    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(int, line)
        y1_full = y1 + roi_start
        y2_full = y2 + roi_start
        length = float(np.hypot(x2 - x1, y2_full - y1_full))
        if length < min_length:
            continue

        width = abs(x2 - x1)
        height = abs(y2_full - y1_full)
        if width < frame_w * 0.08 and height < frame_h * 0.10:
            continue

        angle = abs(np.degrees(np.arctan2(y2_full - y1_full, x2 - x1)))
        grid_like = angle <= 28.0 or angle >= 58.0
        if not grid_like:
            continue

        road_lines.append({
            "color": "green",
            "true_color": "green",
            "detection_type": "road_line",
            "display_label": "GREEN ROAD LINE",
            "x1": x1,
            "y1": y1_full,
            "x2": x2,
            "y2": y2_full,
            "x": int((x1 + x2) / 2),
            "y": int((y1_full + y2_full) / 2),
            "radius": int(max(1.0, length / 2.0)),
            "length": length
        })

    road_center_x = None
    moments = cv2.moments(road_mask)
    if moments["m00"] > 0:
        road_center_x = int(moments["m10"] / moments["m00"])

    return road_lines, road_mask, roi_start, road_center_x

def detect_green_tokens(frame_bright):
    frame_h, frame_w = frame_bright.shape[:2]
    hsv_bright = cv2.cvtColor(frame_bright, cv2.COLOR_BGR2HSV)

    green_mask_1 = cv2.inRange(
        hsv_bright,
        np.array((32, 22, 65), dtype=np.uint8),
        np.array((100, 255, 255), dtype=np.uint8)
    )
    green_mask_2 = cv2.inRange(
        hsv_bright,
        np.array((36, 18, 55), dtype=np.uint8),
        np.array((92, 255, 255), dtype=np.uint8)
    )
    green_mask = cv2.bitwise_or(green_mask_1, green_mask_2)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    green_tokens = []

    token_min_area = max(80.0, frame_w * frame_h * 0.00018)
    token_max_area = frame_w * frame_h * 0.10

    def max_green_token_radius_for_y(center_y):
        y_ratio = float(center_y) / float(max(frame_h, 1))
        if y_ratio < GREEN_TOKEN_MIN_Y_RATIO:
            return 0.0
        if y_ratio < 0.50:
            return max(24.0, min(frame_w, frame_h) * 0.12)
        if y_ratio < 0.70:
            return max(32.0, min(frame_w, frame_h) * 0.15)
        return max(28.0, min(frame_w, frame_h) * 0.19)

    def green_ring_coverage(center_x, center_y, radius):
        outer_radius = int(round(radius * 1.45))
        inner_radius = int(round(radius * 1.08))
        if outer_radius <= inner_radius:
            return 0.0

        outer_mask = np.zeros_like(green_mask)
        inner_mask = np.zeros_like(green_mask)
        center = (int(center_x), int(center_y))
        cv2.circle(outer_mask, center, outer_radius, 255, -1)
        cv2.circle(inner_mask, center, inner_radius, 255, -1)
        ring_mask = cv2.bitwise_and(outer_mask, cv2.bitwise_not(inner_mask))
        ring_area = cv2.countNonZero(ring_mask)
        if ring_area <= 0:
            return 0.0
        ring_green_pixels = cv2.countNonZero(cv2.bitwise_and(green_mask, ring_mask))
        return ring_green_pixels / float(ring_area)

    def append_green_token(center_x, center_y, radius, area, detection_type="token"):
        center_x = int(round(center_x))
        center_y = int(round(center_y))
        radius = int(round(radius))

        if center_y < frame_h * GREEN_TOKEN_MIN_Y_RATIO or center_y > frame_h * 0.94:
            return
        if radius > max_green_token_radius_for_y(center_y):
            return
        if green_ring_coverage(center_x, center_y, radius) > 0.52:
            return

        for existing in green_tokens:
            dx = existing["x"] - center_x
            dy = existing["y"] - center_y
            min_gap = max(existing["radius"], radius) * 0.75
            if (dx * dx + dy * dy) <= (min_gap * min_gap):
                return

        green_tokens.append({
            "color": "green",
            "true_color": "green",
            "detection_type": detection_type,
            "display_label": "GREEN TOKEN",
            "x": center_x,
            "y": center_y,
            "radius": radius,
            "lane": None,
            "area": float(area)
        })

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < token_min_area or area > token_max_area:
            continue

        x, y, width, height = cv2.boundingRect(cnt)
        if width == 0 or height == 0:
            continue
        if width < max(10, frame_w * 0.016) or height < max(10, frame_h * 0.016):
            continue

        center_y_from_box = y + (height * 0.5)
        if center_y_from_box < frame_h * GREEN_TOKEN_MIN_Y_RATIO or center_y_from_box > frame_h * 0.94:
            continue
        width_height_ratio = max(width / float(height), height / float(width))
        if width_height_ratio > 2.15:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        (center_x, center_y), radius = cv2.minEnclosingCircle(cnt)
        if radius < max(7.5, min(frame_w, frame_h) * 0.016):
            continue
        if radius > max_green_token_radius_for_y(center_y):
            continue

        fill_ratio = area / (np.pi * radius * radius)

        close_to_car = center_y_from_box > frame_h * 0.70
        inside_road_view = frame_w * 0.12 <= center_x <= frame_w * 0.88
        close_round_token = (
            close_to_car and
            circularity >= 0.48 and
            fill_ratio >= 0.38 and
            radius >= max(12.0, min(frame_w, frame_h) * 0.025)
        )
        if close_to_car and inside_road_view and not close_round_token:
            continue

        if (
            circularity >= 0.42 and
            fill_ratio >= 0.32
        ):
            append_green_token(center_x, center_y, radius, area)

    blurred_green_mask = cv2.medianBlur(green_mask, 5)
    min_circle_radius = max(5, int(min(frame_w, frame_h) * 0.012))
    max_circle_radius = max(min_circle_radius + 2, int(min(frame_w, frame_h) * 0.070))
    circles = cv2.HoughCircles(
        blurred_green_mask,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(18, int(min(frame_w, frame_h) * 0.045)),
        param1=80,
        param2=13,
        minRadius=min_circle_radius,
        maxRadius=max_circle_radius
    )

    if circles is not None:
        for center_x, center_y, radius in np.round(circles[0, :]).astype("int"):
            if center_x < 0 or center_x >= frame_w or center_y < 0 or center_y >= frame_h:
                continue
            if center_y < frame_h * GREEN_TOKEN_MIN_Y_RATIO:
                continue

            circle_area = np.pi * float(radius) * float(radius)
            if circle_area < token_min_area * 0.55 or circle_area > token_max_area:
                continue

            circle_mask = np.zeros_like(green_mask)
            cv2.circle(circle_mask, (int(center_x), int(center_y)), int(radius), 255, -1)
            green_pixels = cv2.countNonZero(cv2.bitwise_and(green_mask, circle_mask))
            coverage = green_pixels / max(circle_area, 1.0)
            if coverage < 0.30:
                continue
            if green_ring_coverage(center_x, center_y, radius) > 0.52:
                continue

            append_green_token(
                center_x,
                center_y,
                radius,
                circle_area,
                detection_type="token_circle_recovery"
            )

    return green_tokens

def detect_green_tokens_and_road_lines(frame_bright):
    return detect_green_tokens(frame_bright), []

def detect_colored_tokens(frame):
    """
    Detect green, yellow, and red circular tokens from the front camera.

    Red tokens are pale pink/red because of their gradient.  They need a
    separate mask made from the original frame.  The red road markings and
    the player's car are normally much more saturated, so an upper saturation
    limit is used to reject them.
    """
    frame_h, frame_w = frame.shape[:2]

    # Gamma correction is useful for distant green and yellow tokens.
    frame_bright = adjust_gamma(frame, gamma=1.5)
    hsv_bright = cv2.cvtColor(frame_bright, cv2.COLOR_BGR2HSV)

    # Use the original image for red. Gamma correction reduces the saturation
    # contrast of the pale red token and can leave only a fragmented contour.
    hsv_original = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    detected_tokens = []
    green_tokens = detect_green_tokens(frame_bright)
    detected_tokens.extend(green_tokens)
    standard_kernel = np.ones((5, 5), np.uint8)

    # -----------------------------------------------------
    # Yellow detection
    # -----------------------------------------------------
    standard_color_ranges = {
        "yellow": [
            ((18, 55, 135), (34, 255, 255)),
            ((16, 45, 120), (36, 255, 255))
        ]
    }

    for color_name, ranges in standard_color_ranges.items():
        color_mask = None

        for lower, upper in ranges:
            current_mask = cv2.inRange(
                hsv_bright,
                np.array(lower, dtype=np.uint8),
                np.array(upper, dtype=np.uint8)
            )
            color_mask = current_mask if color_mask is None else cv2.bitwise_or(color_mask, current_mask)

        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, standard_kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, standard_kernel)

        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100 or area > 300000:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            min_circularity = 0.62
            if circularity < min_circularity:
                continue

            (x, y), radius = cv2.minEnclosingCircle(cnt)
            detected_tokens.append({
                "color": color_name,
                "true_color": color_name,
                "detection_type": "token",
                "display_label": color_name.upper(),
                "x": int(x),
                "y": int(y),
                "radius": int(radius),
                "area": float(area)
            })

    # -----------------------------------------------------
    # Red token detection
    # -----------------------------------------------------
    # OpenCV hue wraps around: red exists near both 0 and 179.
    #
    # The token is a light pink/red gradient. Saturation is deliberately
    # limited to 20..200 to retain the pale token while rejecting strongly
    # saturated red road borders and most of the player's red car.
    red_mask_1 = cv2.inRange(
        hsv_original,
        np.array((0, 20, 145), dtype=np.uint8),
        np.array((20, 200, 255), dtype=np.uint8)
    )
    red_mask_2 = cv2.inRange(
        hsv_original,
        np.array((160, 20, 145), dtype=np.uint8),
        np.array((179, 200, 255), dtype=np.uint8)
    )
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

    # Close first to reconnect the token's gradient. Use a smaller opening
    # kernel afterwards so distant tokens are not erased.
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Scale the minimum area so the detector works at different resolutions.
    red_min_area = max(70.0, frame_w * frame_h * 0.00030)
    red_max_area = frame_w * frame_h * 0.20

    for cnt in red_contours:
        area = cv2.contourArea(cnt)
        if area < red_min_area or area > red_max_area:
            continue

        x, y, width, height = cv2.boundingRect(cnt)
        if height == 0:
            continue

        aspect_ratio = width / float(height)
        if not (0.58 <= aspect_ratio <= 1.55):
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        (center_x, center_y), radius = cv2.minEnclosingCircle(cnt)

        if radius <= 0:
            continue

        circle_fill_ratio = area / (np.pi * radius * radius)

        # These filters retain round red tokens while rejecting thin road
        # markings, roadside arrows and irregular red parts of the car.
        if circularity < 0.46 or circle_fill_ratio < 0.43:
            continue

        # Ignore HUD text at the top and small fragments near the bottom edge.
        if center_y < frame_h * 0.20 or center_y > frame_h * 0.92:
            continue

        detected_tokens.append({
            "color": "red",
            "true_color": "red",
            "detection_type": "token",
            "display_label": "RED",
            "x": int(center_x),
            "y": int(center_y),
            "radius": int(radius),
            "lane": None,
            "area": float(area)
        })

    # Optional tuning window. Uncomment temporarily if you need to inspect
    # which pixels are selected as red:
    # cv2.imshow("Red Token Mask", cv2.resize(red_mask, (640, 480)))
    # cv2.waitKey(1)

    return detected_tokens, []

def draw_detected_tokens(frame, tokens, green_road_lines=None, road_center_x=None, green_target=None, navigation_target=None, navigation_mode="lane following"):
    """
    Draw tokens and the detected road grid without mixing the two detections.
    """
    display_frame = frame.copy()
    frame_h, frame_w = display_frame.shape[:2]

    text_colors = {
        "green": (0, 255, 0),
        "yellow": (0, 255, 255),
        "red": (0, 0, 255),
        "hidden": (255, 255, 255)
    }

    overlay = display_frame.copy()
    road_overlay = display_frame.copy()
    if green_road_lines:
        for road_line in green_road_lines:
            x1 = int(max(0, min(frame_w - 1, road_line.get("x1", 0))))
            y1 = int(max(0, min(frame_h - 1, road_line.get("y1", 0))))
            x2 = int(max(0, min(frame_w - 1, road_line.get("x2", 0))))
            y2 = int(max(0, min(frame_h - 1, road_line.get("y2", 0))))
            cv2.line(road_overlay, (x1, y1), (x2, y2), (0, 255, 60), 3, cv2.LINE_AA)
            cv2.line(display_frame, (x1, y1), (x2, y2), (0, 170, 30), 1, cv2.LINE_AA)

        cv2.addWeighted(road_overlay, 0.55, display_frame, 0.45, 0, display_frame)

    for token in tokens:
        label = token.get("display_label", token.get("display_color", token["color"]))
        if token.get("true_color", token.get("color")) == "green":
            radius = float(token.get("radius", 0))
            area = float(token.get("area", 0))
            fill_ratio = area / (np.pi * radius * radius) if radius > 0 else 0.0
            close_round_token = (
                token["y"] > frame_h * 0.70 and
                radius >= max(12.0, min(frame_w, frame_h) * 0.025) and
                fill_ratio >= 0.38
            )
            if token["y"] > frame_h * 0.70 and frame_w * 0.12 <= token["x"] <= frame_w * 0.88 and not close_round_token:
                continue
        if label == "GREEN TOKEN":
            label = "GREEN"
            color = (0, 255, 0)
        else:
            color = text_colors.get(label, text_colors.get(token["color"], (255, 255, 255)))
        center = (token["x"], token["y"])
        radius = token["radius"]
        # Semi-transparent filled circle
        cv2.circle(overlay, center, radius, color, -1)
        # Bounding box
        x1 = max(center[0]-radius, 0)
        y1 = max(center[1]-radius, 0)
        x2 = min(center[0]+radius, frame_w-1)
        y2 = min(center[1]+radius, frame_h-1)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        # Color label
        cv2.putText(display_frame, label.upper(), (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Blend overlay with original frame
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, display_frame, 1-alpha, 0, display_frame)
    return display_frame

def scale_token_detections(tokens, scale):
    if scale == 1.0:
        return tokens

    inverse_scale = 1.0 / scale
    inverse_area_scale = inverse_scale * inverse_scale
    scaled_tokens = []
    for token in tokens:
        scaled_token = dict(token)
        for key in ("x", "y", "radius"):
            if key in scaled_token:
                scaled_token[key] = int(round(scaled_token[key] * inverse_scale))
        if "area" in scaled_token:
            scaled_token["area"] = float(scaled_token["area"] * inverse_area_scale)
        scaled_tokens.append(scaled_token)
    return scaled_tokens

def scale_road_detections(road_lines, road_mask, roi_start, scale, full_frame_shape):
    full_h, full_w = full_frame_shape[:2]
    if scale == 1.0:
        return road_lines, road_mask, roi_start, None

    inverse_scale = 1.0 / scale
    scaled_lines = []
    for road_line in road_lines:
        scaled_line = dict(road_line)
        for key in ("x1", "y1", "x2", "y2", "x", "y", "radius"):
            if key in scaled_line:
                scaled_line[key] = int(round(scaled_line[key] * inverse_scale))
        if "length" in scaled_line:
            scaled_line["length"] = float(scaled_line["length"] * inverse_scale)
        scaled_lines.append(scaled_line)

    roi_start_full = int(round(roi_start * inverse_scale))
    roi_start_full = max(0, min(full_h - 1, roi_start_full))
    road_mask_full = None
    road_center_x_full = None
    if road_mask is not None:
        roi_height_full = max(1, full_h - roi_start_full)
        road_mask_full = np.zeros((full_h, full_w), dtype=np.uint8)
        road_mask_full[roi_start_full:, :] = cv2.resize(
            road_mask,
            (full_w, roi_height_full),
            interpolation=cv2.INTER_NEAREST
        )
        moments = cv2.moments(road_mask_full)
        if moments["m00"] > 0:
            road_center_x_full = int(moments["m10"] / moments["m00"])

    return scaled_lines, road_mask_full, roi_start_full, road_center_x_full


def lane_from_x(x, y=None, frame_width=640, frame_height=480):
    # If y is not provided, fallback to the old simple split method
    if y is None:
        lane_count = 5
        lane_width = frame_width / lane_count
        lane = int(x // lane_width) - 2
        return max(-2, min(2, lane))

    # Distant tokens sit close to the horizon, where tiny y changes make the
    # perspective slope explode. Use stable horizontal bands there.
    if y <= frame_height * 0.62:
        lane_count = 5
        lane_width = frame_width / lane_count
        lane = int(x // lane_width) - 2
        return max(-2, min(2, lane))
        
    # Perspective-based lane calculation
    # The road lines radiate from a vanishing point near the horizon
    vp_x = frame_width / 2.0
    vp_y = frame_height * 0.45  # Approximate horizon line
    
    if y <= vp_y:
        lane_count = 5
        lane_width = frame_width / lane_count
        lane = int(x // lane_width) - 2
        return max(-2, min(2, lane))
        
    dy = y - vp_y
    dx = x - vp_x
    slope = dx / dy
    
    # Slopes defining the lane boundaries
    if slope < -0.6:
        return -2
    elif slope < -0.2:
        return -1
    elif slope < 0.2:
        return 0
    elif slope < 0.6:
        return 1
    else:
        return 2


def hazard_blocks_green_target(hazard, green_target, frame_width=640, frame_height=480):
    if hazard is None or green_target is None:
        return False

    if hazard.get('lane') == green_target.get('lane'):
        return True

    x_gap = abs(float(hazard.get('x', 0)) - float(green_target.get('x', 0)))
    y_margin = frame_height * GREEN_HAZARD_BLOCK_Y_MARGIN_RATIO
    return (
        x_gap <= frame_width * GREEN_HAZARD_BLOCK_X_RATIO and
        float(hazard.get('y', 0)) >= float(green_target.get('y', 0)) - y_margin
    )


def detect_trailing_and_police(back_frame):
    """
    Robust vision-based detection using the back camera frame.
    - trailing_detected: True when a large moving object is detected directly behind the player in a constrained ROI.
    - police_detected: True when bright neon red+blue lights are detected horizontally aligned in the back camera.

    Returns: (trailing_detected(bool), police_detected(bool), largest_area(float), largest_bbox(tuple|None))
    """
    if back_frame is None:
        return False, False, 0.0, None

    frame_h, frame_w = back_frame.shape[:2]

    # --- 1. Trailing Car Detection (Constrained ROI) ---
    # We only look at the center lane directly behind us to ignore the fast-moving scenery on the sides.
    # X: 35% to 65% of width, Y: 40% to 90% of height.
    t_roi_x1 = int(frame_w * 0.35)
    t_roi_x2 = int(frame_w * 0.65)
    t_roi_y1 = int(frame_h * 0.40)
    t_roi_y2 = int(frame_h * 0.90)

    t_roi = back_frame[t_roi_y1:t_roi_y2, t_roi_x1:t_roi_x2]
    gray_roi = cv2.cvtColor(t_roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

    with data_lock:
        prev_gray_roi = shared_data.get('back_prev_gray')
        prev_area = shared_data.get('back_prev_area', 0.0)

    largest_area = 0.0
    largest_bbox = None
    trailing = False

    if prev_gray_roi is not None and prev_gray_roi.shape == gray_roi.shape:
        # Frame differencing purely inside the safe ROI
        diff = cv2.absdiff(prev_gray_roi, gray_roi)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Tokens lying on the ground appear as very wide, flat objects when they pass under the car.
            # A trailing car is a 3D object and will have a much taller bounding box.
            aspect_ratio = float(w) / h if h > 0 else 0.0
            
            # Filter out flat objects (tokens) and require a minimum physical height in the ROI
            if aspect_ratio < 2.5 and h > (t_roi_y2 - t_roi_y1) * 0.15:
                if area > largest_area:
                    largest_area = area
                    # Offset bounding box back to original frame coordinates
                    largest_bbox = (x + t_roi_x1, y + t_roi_y1, w, h)

        # Minimum threshold for an approaching car within this ROI (around 6000 pixels to ignore small tokens)
        TRAILING_MOTION_AREA_MIN_ROI = 6000
        TRAILING_APPROACH_RATIO = 1.05  # slightly more forgiving since ROI is smaller

        if largest_area > TRAILING_MOTION_AREA_MIN_ROI:
            if prev_area > 0 and largest_area > (prev_area * TRAILING_APPROACH_RATIO):
                trailing = True

    # --- 2. Police Detection (Bright Neon Color Filters) ---
    # Police cars have bright lightbars. We look at the upper-middle section.
    p_roi_x1 = int(frame_w * 0.20)
    p_roi_x2 = int(frame_w * 0.80)
    p_roi_y1 = int(frame_h * 0.30)
    p_roi_y2 = int(frame_h * 0.70)
    
    p_roi = back_frame[p_roi_y1:p_roi_y2, p_roi_x1:p_roi_x2]
    hsv_roi = cv2.cvtColor(p_roi, cv2.COLOR_BGR2HSV)

    # Use very strict Saturation (>180) and Value (>200) thresholds to ignore dull environmental colors
    lower_red1 = np.array([0, 180, 200])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 180, 200])
    upper_red2 = np.array([180, 255, 255])
    
    lower_blue = np.array([100, 180, 200])
    upper_blue = np.array([140, 255, 255])

    red_mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    blue_mask = cv2.inRange(hsv_roi, lower_blue, upper_blue)

    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    police = False
    POLICE_MIN_BRIGHT_AREA = 100

    red_centers = []
    for cnt in red_contours:
        if cv2.contourArea(cnt) > POLICE_MIN_BRIGHT_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            red_centers.append((x + w/2.0, y + h/2.0))

    blue_centers = []
    for cnt in blue_contours:
        if cv2.contourArea(cnt) > POLICE_MIN_BRIGHT_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            blue_centers.append((x + w/2.0, y + h/2.0))

    if red_centers and blue_centers:
        # Check if the bright red and blue lights are horizontally aligned (lightbar on top of the car)
        # Max horizontal distance and max vertical distance
        max_x_dist = frame_w * 0.40
        max_y_dist = frame_h * 0.10  # They should be relatively on the same horizontal plane

        for rx, ry in red_centers:
            for bx, by in blue_centers:
                if abs(rx - bx) <= max_x_dist and abs(ry - by) <= max_y_dist:
                    police = True
                    break
            if police:
                break

    # Save state for next frame
    with data_lock:
        shared_data['back_prev_gray'] = gray_roi
        shared_data['back_prev_area'] = largest_area

    return trailing, police, largest_area, largest_bbox

def get_active_yellow_effect():
    now = time.time()
    with data_lock:
        effect = shared_data.get('active_yellow_effect')
        effect_until = shared_data.get('yellow_effect_until', 0.0)
        if effect and now >= effect_until:
            shared_data['active_yellow_effect'] = None
            shared_data['yellow_effect_until'] = 0.0
            return None
        return effect

def start_random_yellow_effect():
    effect = random.choice(YELLOW_EFFECTS)
    now = time.time()
    with data_lock:
        if shared_data.get('active_yellow_effect') and now < shared_data.get('yellow_effect_until', 0.0):
            return
        shared_data['active_yellow_effect'] = effect
        shared_data['yellow_effect_until'] = now + YELLOW_EFFECT_DURATION_SECONDS
        shared_data['run_summary']['yellow_effects'][effect] += 1
        if effect == 'hide_next_token_type':
            shared_data['hidden_next_token_type'] = True

def maybe_record_collected_token(tokens, frame_width, frame_height):
    global locked_green_lane, locked_green_until, locked_green_target, locked_green_target_until, last_token_lane, last_token_time, committed_target_lane, committed_target_until, committed_target_reason

    now = time.time()
    collection_y = frame_height * TOKEN_COLLECTION_Y_RATIO
    collected = None

    for token in tokens:
        if token['y'] < collection_y:
            continue

        token_lane = lane_from_x(token['x'], token['y'], frame_width=frame_width, frame_height=frame_height)
        if token_lane != current_lane:
            continue

        color = token.get('true_color', token.get('color'))
        if color not in ['green', 'yellow', 'red']:
            continue

        token_id = f"{token_lane}:{color}"

        with data_lock:
            recent_collections = shared_data.get('last_collected_by_lane_color', {})
            recent_same_token = (
                now - recent_collections.get(token_id, 0.0) < TOKEN_COLLECTION_COOLDOWN_SECONDS
            )

        if recent_same_token:
            continue

        collected = (token_id, color)
        break

    if collected is None:
        return

    token_id, color = collected
    with data_lock:
        shared_data['last_collected_line_id'] = token_id
        shared_data['last_collected_time'] = now
        shared_data['last_collected_by_lane_color'][token_id] = now
        shared_data['run_summary'][f'{color}_collected'] += 1

    if color == 'yellow':
        start_random_yellow_effect()
    elif color == 'green':
        locked_green_lane = None
        locked_green_until = 0.0
        locked_green_target = None
        locked_green_target_until = 0.0
        last_token_lane = None
        last_token_time = 0.0
        committed_target_lane = None
        committed_target_until = 0.0
        committed_target_reason = "MAINTAIN"

def apply_camera_effects(front_frame):
    effect = get_active_yellow_effect()
    now = time.time()

    if not front_camera_delay_buffer or now - front_camera_delay_buffer[-1][0] >= 0.05:
        front_camera_delay_buffer.append((now, front_frame.copy()))
    while front_camera_delay_buffer and now - front_camera_delay_buffer[0][0] > CAMERA_DELAY_SECONDS + 1.0:
        front_camera_delay_buffer.pop(0)

    if effect == 'camera_input_delay':
        delayed_frame = front_camera_delay_buffer[0][1]
        for timestamp, buffered_frame in front_camera_delay_buffer:
            delayed_frame = buffered_frame
            if now - timestamp <= CAMERA_DELAY_SECONDS:
                break
        return delayed_frame.copy()

    if effect == 'corrupted_camera_input':
        noise = np.random.normal(0, 45, front_frame.shape).astype(np.int16)
        corrupted = np.clip(front_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return corrupted

    return front_frame

def apply_token_visibility_effects(tokens):
    effect = get_active_yellow_effect()

    if effect == 'tokens_invisible':
        return []

    with data_lock:
        hide_next = shared_data.get('hidden_next_token_type', False)

    if not hide_next or not tokens:
        return tokens

    hidden_tokens = [dict(token) for token in tokens]
    closest_index = max(range(len(hidden_tokens)), key=lambda idx: hidden_tokens[idx]['y'])
    hidden_tokens[closest_index]['display_color'] = 'hidden'

    with data_lock:
        shared_data['hidden_next_token_type'] = False

    return hidden_tokens

def send_control_packet(steering_input, acceleration_input):
    global control_conn

    data = struct.pack('ff', float(steering_input), float(acceleration_input))
    effect = get_active_yellow_effect()

    if effect != 'action_output_delay':
        action_delay_buffer.clear()
        control_conn.sendall(data)
        return

    now = time.time()
    action_delay_buffer.append((now, data))
    while action_delay_buffer and now - action_delay_buffer[0][0] > ACTION_DELAY_SECONDS + 1.0:
        action_delay_buffer.pop(0)

    delayed_data = action_delay_buffer[0][1]
    for timestamp, buffered_data in action_delay_buffer:
        delayed_data = buffered_data
        if now - timestamp <= ACTION_DELAY_SECONDS:
            break
    control_conn.sendall(delayed_data)

def processing_task():
    #This is where you write your image processing code to decide how to control the car
    #You can use libraries like OpenCV to process the image
    #There is no limtation to the complexity of the processing task, you can use any libraries you want
    #Remember to use the shared_data to get the latest frame
    if not hasattr(processing_task, "frame_index"):
        processing_task.frame_index = 0
        processing_task.cached_road_result = ([], None, 0, None)

    task_start = time.perf_counter()
    with data_lock:
        front_frame = shared_data['latest_front_frame']
    
    with data_lock:
        back_frame = shared_data.get('latest_back_frame')

    if front_frame is not None:
        processing_task.frame_index += 1
        processed_front_frame = apply_camera_effects(front_frame)
        token_frame = cv2.resize(
            processed_front_frame,
            None,
            fx=TOKEN_PROCESS_SCALE,
            fy=TOKEN_PROCESS_SCALE,
            interpolation=cv2.INTER_LINEAR
        )
        tokens, _ = detect_colored_tokens(token_frame)
        tokens = scale_token_detections(tokens, TOKEN_PROCESS_SCALE)
        tokens = apply_token_visibility_effects(tokens)
        green_token_count = sum(1 for token in tokens if token.get('true_color', token.get('color')) == 'green')
        token_exclusion_mask = build_token_exclusion_mask(tokens, processed_front_frame.shape)

        previous_processing_ms = shared_data.get('processing_ms', 0.0)
        road_due = (
            previous_processing_ms <= ROAD_SKIP_IF_SLOW_MS and
            (
                processing_task.frame_index % ROAD_DETECTION_STRIDE == 0 or
                processing_task.cached_road_result[1] is None
            )
        )

        if road_due:
            road_frame = cv2.resize(
                processed_front_frame,
                None,
                fx=ROAD_PROCESS_SCALE,
                fy=ROAD_PROCESS_SCALE,
                interpolation=cv2.INTER_LINEAR
            )
            token_exclusion_small = cv2.resize(
                token_exclusion_mask,
                (road_frame.shape[1], road_frame.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            green_road_lines_small, road_mask_small, roi_start_small, _ = detect_road_lines(road_frame, token_exclusion_small)
            green_road_lines, road_mask, roi_start, road_center_x = scale_road_detections(
                green_road_lines_small,
                road_mask_small,
                roi_start_small,
                ROAD_PROCESS_SCALE,
                processed_front_frame.shape
            )
            processing_task.cached_road_result = (green_road_lines, road_mask, roi_start, road_center_x)
        else:
            green_road_lines, road_mask, roi_start, road_center_x = processing_task.cached_road_result

        with data_lock:
            shared_data['detected_tokens'] = tokens
            shared_data['green_token_count'] = green_token_count
            if green_token_count > 0:
                shared_data['last_green_detected_time'] = time.time()
            shared_data['detected_green_road_lines'] = green_road_lines
            shared_data['road_center_x'] = road_center_x
            selected_green_target = shared_data.get('selected_green_target')
            selected_green_target_type = shared_data.get('selected_green_target_type', 'NONE')
            navigation_target_x = shared_data.get('navigation_target_x')
            navigation_mode = shared_data.get('navigation_mode', 'lane following')

        debug_frame = draw_detected_tokens(
            processed_front_frame,
            tokens,
            green_road_lines=green_road_lines,
            road_center_x=road_center_x,
            green_target=selected_green_target,
            navigation_target={'x': navigation_target_x} if navigation_target_x is not None else None,
            navigation_mode=navigation_mode,
        )

        proc_ms = (time.perf_counter() - task_start) * 1000.0
        proc_fps = 1000.0 / proc_ms if proc_ms > 0 else 0.0
        with data_lock:
            shared_data['processing_ms'] = proc_ms
            shared_data['processing_fps'] = proc_fps
            steering_dbg = shared_data.get('steering_input', 0.0)
            accel_dbg = shared_data.get('acceleration_input', 0.0)
            decision_debug = shared_data.get('decision_debug', '')
            slow_reason = shared_data.get('slow_reason', 'none')
            last_green_detected_time = shared_data.get('last_green_detected_time', 0.0)
            green_target_source = shared_data.get('green_target_source', 'none')
            hazard_debug = shared_data.get('hazard_debug', 'none')
        token_count = len(tokens)
        road_line_count = len(green_road_lines)
        green_age = time.time() - last_green_detected_time if last_green_detected_time > 0 else None

        cv2.putText(debug_frame, f"Mode: {navigation_mode}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(debug_frame, f"Target: {selected_green_target_type}", (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(debug_frame, f"Tokens: {token_count} | Roads: {road_line_count}", (10, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(debug_frame, f"Steering: {steering_dbg:+.2f} | Accel: {accel_dbg:.2f}", (10, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(debug_frame, f"Proc: {proc_ms:.1f} ms | FPS: {proc_fps:.1f}", (10, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(debug_frame, f"Speed: {slow_reason}", (10, 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(debug_frame, f"GreenCount: {green_token_count} | GreenAge: {'none' if green_age is None else f'{green_age:.2f}s'} | Source: {green_target_source}", (10, 148),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(debug_frame, hazard_debug[:120], (10, 168),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(debug_frame, decision_debug[:120], (10, 188),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        debug_frame = cv2.resize(debug_frame, (640, 480))
        cv2.imshow("Front Camera", debug_frame)
        cv2.waitKey(1)

    # Run trailing / police detection on back camera
    if back_frame is not None:
        trailing, police, area, bbox = detect_trailing_and_police(back_frame)
        with data_lock:
            shared_data['trailing_detected'] = trailing
            shared_data['police_detected'] = police
            shared_data['danger_detected'] = trailing or False
            
            if police:
                shared_data['run_summary']['police_appeared'] = True
            if trailing:
                shared_data['run_summary']['trailing_appeared'] = True

        # Show the rear view in the same window, with detection overlays on top.
        try:
            dbg = back_frame.copy()
            if bbox is not None and trailing:
                x, y, w, h = bbox
                cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 255), 2)

            status_color = (0, 255, 0)
            if trailing:
                status_color = (0, 255, 255)
            if police:
                status_color = (0, 0, 255)

            lines = [
                "Back Camera",
                f"Trailing: {trailing}",
                f"Police: {police}",
                f"Area: {int(area)}"
            ]
            for idx, line in enumerate(lines):
                cv2.putText(dbg, line, (10, 30 + idx * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            dbg = cv2.resize(dbg, (640, 480))
            cv2.imshow("Back Camera", dbg)
            cv2.waitKey(1)
        except Exception:
            pass

# def send_controls_task():
#     #This is where you send the control commands to the car using the control_conn
#     global control_conn
#     if control_conn is None:
#         return
    
#     #these are the variables used to control the car
#     #steering_input: -1.0 to 1.0 (left to right)
#     #acceleration_input: -1.0 to 1.0 (reverse to forward)
#     #this example always accelerate forward
#     steering_input = 0.0
#     acceleration_input = 1.0

#     try:
#         # Pack and send the control command
#         data = struct.pack('ff', steering_input, acceleration_input)
#         control_conn.sendall(data)
#     except Exception as e:
#         print(f"Control send error: {e}")
#         control_conn = None
# Global variables to track our steering state machine
steering_state = 0       # 0 = Idle, 1 = Tapping, 2 = Resetting
tap_loop_count = 0       # Counts how many loops we hold the steering wheel
current_lane = START_LANE # Tracks our car's actual lane position (-2, -1, 0, 1, 2)
last_token_lane = None   # Keeps the last good token decision when tokens briefly disappear
last_token_time = 0.0
locked_green_lane = None
locked_green_until = 0.0
locked_green_target = None
locked_green_target_until = 0.0
committed_target_lane = None
committed_target_until = 0.0
committed_target_reason = "MAINTAIN"
avoidance_lock_until = 0.0
avoidance_lock_direction = 0.0
avoidance_lock_hazard = None
run_start_time = None
def send_controls_task():
    global control_conn, steering_state, tap_loop_count, current_lane, last_token_lane, last_token_time, locked_green_lane, locked_green_until, locked_green_target, locked_green_target_until, committed_target_lane, committed_target_until, committed_target_reason, avoidance_lock_until, avoidance_lock_direction, avoidance_lock_hazard, run_start_time
    
    if control_conn is None:
        return
    
    with data_lock:
        tokens_snapshot = list(shared_data.get('detected_tokens', []))
        green_road_lines_snapshot = list(shared_data.get('detected_green_road_lines', []))
        road_center_x = shared_data.get('road_center_x')
        front_frame = shared_data.get('latest_front_frame')
        run_summary = shared_data.get('run_summary', {})
        green_streak = run_summary.get('green_collected', 0)
        red_streak = run_summary.get('red_collected', 0)

    steering_input = 0.0
    acceleration_input = CAR_ACCELERATION 

    if run_start_time is not None and time.time() - run_start_time < START_CENTER_HOLD_SECONDS:
        try: control_conn.sendall(struct.pack('ff', 0.0, 0.0))
        except Exception: pass
        current_lane = START_LANE
        return

    img_w, img_h = (front_frame.shape[1], front_frame.shape[0]) if front_frame is not None else (640, 480)
    maybe_record_collected_token(tokens_snapshot, img_w, img_h)

    with data_lock:
        run_summary = shared_data.get('run_summary', {})
        green_streak = run_summary.get('green_collected', 0)
        red_streak = run_summary.get('red_collected', 0)

    # 1. SPEED MODIFIER & TUNING PARAMETERS
    # =========================================================
    estimated_speed_modifier = 1.0 + (green_streak * 0.10) - (red_streak * 0.20)
    estimated_speed_modifier = max(0.70, estimated_speed_modifier)

    acceleration_input = max(
        MIN_ACCELERATION_WHEN_SLOWED,
        min(1.0, CAR_ACCELERATION)
    )

    adaptive_tap_loops = max(
        28,
        int(LANE_CHANGE_TAP_LOOPS * (estimated_speed_modifier ** 0.35))
    )

    adaptive_reset_loops = max(
        4,
        int(LANE_CHANGE_RESET_LOOPS / (estimated_speed_modifier ** 0.10))
    )

    # =========================================================
    # 2. TOKEN PRIORITY DECISION
    # =========================================================
    best_green = None
    any_green_visible = False
    any_green_road_line_visible = bool(green_road_lines_snapshot)
    visible_hazard_lanes = set()
    blocking_hazard_lanes = set()
    closest_hazard = None
    direct_hazard = None
    green_y_by_lane = {}
    green_target_by_lane = {}
    active_green_target = None
    car_center_x = img_w / 2.0

    for t in tokens_snapshot:
        color = t.get('color')
        if color not in ['green', 'red', 'yellow'] or t['y'] < TOKEN_DECISION_Y_MIN:
            continue

        t_lane = lane_from_x(t['x'], t['y'], frame_width=img_w, frame_height=img_h)
        t_lane = max(-2, min(2, t_lane))

        if color in ['red', 'yellow'] and t['y'] >= HAZARD_AVOID_Y_MIN:
            visible_hazard_lanes.add(t_lane)
            center_error_px = float(t['x']) - car_center_x
            center_error = abs(center_error_px / max(car_center_x, 1.0))
            y_ratio = float(t['y']) / float(max(img_h, 1))
            blocks_current_lane = t_lane == current_lane
            blocks_forward_path = center_error < HAZARD_DIRECT_CENTER_RATIO and y_ratio >= HAZARD_DIRECT_Y_RATIO
            if blocks_current_lane or blocks_forward_path:
                blocking_hazard_lanes.add(t_lane)
                if closest_hazard is None or t['y'] > closest_hazard['y']:
                    closest_hazard = {'lane': t_lane, 'x': t['x'], 'y': t['y'], 'color': color, 'error': center_error_px, 'direct': blocks_forward_path}
            if blocks_forward_path and (direct_hazard is None or t['y'] > direct_hazard['y']):
                direct_hazard = {'lane': t_lane, 'x': t['x'], 'y': t['y'], 'color': color, 'error': center_error_px, 'direct': True}
            continue

        if color != 'green':
            continue

        any_green_visible = True
        green_y_by_lane[t_lane] = max(green_y_by_lane.get(t_lane, 0), t['y'])
        lane_green_target = green_target_by_lane.get(t_lane)
        if lane_green_target is None or t['y'] > lane_green_target['y']:
            green_target_by_lane[t_lane] = {'lane': t_lane, 'x': t['x'], 'y': t['y']}
        lane_gap = abs(t_lane - current_lane)
        center_error = abs((t['x'] - car_center_x) / max(car_center_x, 1.0))
        candidate = {
            'lane': t_lane,
            'x': t['x'],
            'y': t['y'],
            'score': t['y'] - (lane_gap * img_h * 0.15) - (center_error * img_h * 0.10)
        }
        if lane_gap == 0:
            candidate['score'] += img_h * 0.12
        if center_error < 0.12:
            candidate['score'] += img_h * 0.08

        if best_green is None or candidate['score'] > best_green['score']:
            best_green = candidate

    tokens_visible = any_green_visible or bool(visible_hazard_lanes)
    if estimated_speed_modifier > 1.25:
        acceleration_input = min(acceleration_input, 1.0)

    # =========================================================
    # 3. GREEN-FIRST TARGET DECISION
    # =========================================================
    chosen_target_lane = None
    decision_reason = "MAINTAIN"
    green_debug = "none"
    green_target_source = "none"

    now = time.time()
    if direct_hazard is not None:
        avoidance_lock_direction = 1.0 if direct_hazard['error'] <= 0 else -1.0
        avoidance_lock_until = now + HAZARD_DECISION_LOCK_SECONDS
        avoidance_lock_hazard = dict(direct_hazard)

    avoidance_lock_active = now < avoidance_lock_until and avoidance_lock_direction != 0.0
    if avoidance_lock_active and avoidance_lock_hazard is not None:
        closest_hazard = dict(avoidance_lock_hazard)
        blocking_hazard_lanes.add(closest_hazard['lane'])

    green_clear_of_direct_hazard = (
        best_green is not None and
        best_green['lane'] not in blocking_hazard_lanes and
        not hazard_blocks_green_target(direct_hazard, best_green, img_w, img_h) and
        not hazard_blocks_green_target(avoidance_lock_hazard if avoidance_lock_active else None, best_green, img_w, img_h)
    )

    if green_clear_of_direct_hazard:
        direct_hazard = None
        if avoidance_lock_active:
            avoidance_lock_until = 0.0
            avoidance_lock_direction = 0.0
            avoidance_lock_hazard = None
            avoidance_lock_active = False
            closest_hazard = None
            blocking_hazard_lanes = {
                lane for lane in blocking_hazard_lanes
                if lane == best_green['lane']
            }

    committed_green_active = committed_target_reason in ["COLLECT_GREEN", "GREEN_LOCK", "WAIT_GREEN"]
    committed_green_visible = (
        committed_green_active and
        committed_target_lane is not None and
        committed_target_lane not in blocking_hazard_lanes and
        green_y_by_lane.get(committed_target_lane, 0) > 0
    )

    if green_clear_of_direct_hazard:
        active_green_target = best_green
        chosen_target_lane = best_green['lane']
        green_debug = f"lane {best_green['lane']} x {best_green['x']} y {best_green['y']}"
        green_target_source = "current detection"
        locked_green_lane = chosen_target_lane
        locked_green_until = now + GREEN_TARGET_LOCK_SECONDS
        locked_green_target = dict(best_green)
        locked_green_target_until = now + GREEN_TARGET_LOCK_SECONDS
        decision_reason = "COLLECT_GREEN"
        last_token_lane = chosen_target_lane
        last_token_time = now
    elif avoidance_lock_active or direct_hazard is not None:
        decision_reason = "AVOID_RED_YELLOW"
        chosen_target_lane = current_lane + (1 if avoidance_lock_direction > 0 else -1)
        chosen_target_lane = max(-2, min(2, chosen_target_lane))
        green_debug = "blocked by direct hazard"
    elif committed_green_visible:
        active_green_target = green_target_by_lane.get(committed_target_lane)
        chosen_target_lane = committed_target_lane
        green_debug = f"committed lane {committed_target_lane} x {active_green_target['x']} y {green_y_by_lane[committed_target_lane]}"
        green_target_source = "current detection"
        decision_reason = "GREEN_LOCK"
    elif locked_green_target is not None and now < locked_green_target_until:
        active_green_target = locked_green_target
        chosen_target_lane = locked_green_lane if locked_green_lane is not None else current_lane
        green_debug = f"locked token lane {active_green_target['lane']} x {active_green_target['x']} y {active_green_target['y']}"
        green_target_source = "memory lock"
        decision_reason = "GREEN_LOCK"
    elif locked_green_lane is not None and now < locked_green_until and locked_green_lane not in blocking_hazard_lanes:
        chosen_target_lane = locked_green_lane
        green_target_source = "memory lock"
        decision_reason = "GREEN_LOCK"
    elif locked_green_lane is not None and locked_green_lane in blocking_hazard_lanes:
        locked_green_lane = None
        locked_green_until = 0.0
        locked_green_target = None
        locked_green_target_until = 0.0
        decision_reason = "AVOID_RED_YELLOW"

    if not any_green_visible:
        last_token_lane = None
        last_token_time = 0.0
    if best_green is None or best_green['lane'] in blocking_hazard_lanes:
        if not committed_green_visible:
            committed_target_lane = None
            committed_target_until = 0.0
            committed_target_reason = "MAINTAIN"

    green_target_active = decision_reason in ["COLLECT_GREEN", "GREEN_LOCK"]
    current_lane_blocked = current_lane in blocking_hazard_lanes
    target_lane_blocked = chosen_target_lane in blocking_hazard_lanes if chosen_target_lane is not None else False

    if target_lane_blocked or (closest_hazard is not None and not green_target_active):
        clear_lanes = [lane for lane in [-2, -1, 0, 1, 2] if lane not in blocking_hazard_lanes]
        if clear_lanes:
            if closest_hazard is not None:
                hazard_lane = closest_hazard['lane']
                chosen_target_lane = min(
                    clear_lanes,
                    key=lambda lane: (abs(lane - current_lane), -abs(lane - hazard_lane), abs(lane))
                )
            else:
                chosen_target_lane = min(clear_lanes, key=lambda lane: (abs(lane - current_lane), abs(lane)))
            decision_reason = "AVOID_RED_YELLOW"

    # Baseline Strategy
    if chosen_target_lane is None:
        chosen_target_lane = current_lane
        decision_reason = "MAINTAIN"

    navigation_target = None
    selected_green_target_type = "NONE"
    selected_target_reason = "MAINTAIN"
    slow_reason = "speed rule"
    car_center_x = img_w / 2.0
    green_x = None
    green_y = None
    green_error = None
    hazard_x = None
    hazard_y = None
    hazard_error = None

    blocking_hazard = closest_hazard if closest_hazard is not None and closest_hazard['lane'] in blocking_hazard_lanes else None

    if active_green_target is not None and active_green_target['lane'] not in blocking_hazard_lanes:
        navigation_target = dict(active_green_target)
        selected_green_target_type = "GREEN TOKEN"
        selected_target_reason = "COLLECT_GREEN"
        green_x = float(active_green_target['x'])
        green_y = float(active_green_target['y'])
        green_error = green_x - car_center_x
    elif blocking_hazard is not None:
        navigation_target = dict(blocking_hazard)
        selected_green_target_type = "HAZARD AVOID"
        selected_target_reason = "AVOID_RED_YELLOW"
        hazard_x = float(blocking_hazard['x'])
        hazard_y = float(blocking_hazard['y'])
        hazard_error = hazard_x - car_center_x
    elif road_center_x is not None:
        navigation_target = {
            'x': int(road_center_x),
            'y': int(img_h * 0.35)
        }
        selected_green_target_type = "ROAD PATH"
        selected_target_reason = "MAINTAIN"

    if decision_reason == "MAINTAIN":
        committed_target_lane = None
        committed_target_until = 0.0
        committed_target_reason = "MAINTAIN"
    else:
        committed_target_lane = chosen_target_lane
        committed_target_reason = decision_reason
        lock_seconds = HAZARD_DECISION_LOCK_SECONDS if decision_reason == "AVOID_RED_YELLOW" else DECISION_LOCK_SECONDS
        committed_target_until = now + lock_seconds

    # =========================================================
    # 4. DIRECT STEERING CONTROL
    # =========================================================
    if navigation_target is not None:
        target_x = float(navigation_target['x'])

        if selected_green_target_type == "HAZARD AVOID":
            hazard_error = target_x - car_center_x
            if avoidance_lock_active:
                steering_input = max(-1.0, min(1.0, avoidance_lock_direction * HAZARD_AVOID_STEER))
            else:
                hazard_sign = 1.0 if hazard_error > 0 else -1.0
                steering_input = max(-1.0, min(1.0, -hazard_sign * HAZARD_AVOID_STEER))
            selected_target_reason = "AVOID_RED_YELLOW"
            slow_reason = "speed rule"

        elif selected_green_target_type == "GREEN TOKEN":
            green_error = target_x - car_center_x
            gain = 4.8
            steering_input = max(-1.0, min(1.0, (green_error / max(car_center_x, 1.0)) * gain))
            if abs(green_error) < car_center_x * 0.025:
                steering_input = 0.0
            elif abs(steering_input) < 0.28:
                steering_input = 0.28 if steering_input > 0 else -0.28
            selected_target_reason = "COLLECT_GREEN"
            slow_reason = "speed rule"

        else:
            road_error = target_x - car_center_x
            steering_input = max(-0.75, min(0.75, (road_error / max(car_center_x, 1.0)) * 1.7))
            selected_target_reason = "MAINTAIN"
            slow_reason = "speed rule"

    else:
        steering_input = 0.0
        selected_target_reason = "MAINTAIN"
        slow_reason = "speed rule"

    if navigation_target is not None:
        navigation_target_x = int(round(float(navigation_target['x'])))
        navigation_target_y = int(round(float(navigation_target.get('y', img_h * 0.35))))
    else:
        navigation_target_x = None
        navigation_target_y = None

    if selected_green_target_type == "HAZARD AVOID":
        navigation_mode = "hazard avoidance"
    elif selected_green_target_type == "GREEN TOKEN":
        navigation_mode = "green token chasing"
    else:
        navigation_mode = "lane following"

    with data_lock:
        hazard_debug = "none" if closest_hazard is None else f"{closest_hazard['color']} lane {closest_hazard['lane']} y {closest_hazard['y']}"
        green_token_count = sum(1 for token in tokens_snapshot if token.get('true_color') == 'green')
        selected_green_x = None if active_green_target is None else active_green_target.get('x')
        selected_green_y = None if active_green_target is None else active_green_target.get('y')
        current_hazard_x = hazard_x
        current_hazard_y = hazard_y
        current_hazard_error = hazard_error
        avoidance_direction_label = "none"
        if selected_green_target_type == "HAZARD AVOID":
            direction_value = avoidance_lock_direction if avoidance_lock_active else steering_input
            avoidance_direction_label = "right" if direction_value > 0 else "left"
        shared_data['selected_green_target_type'] = selected_green_target_type
        shared_data['selected_green_target'] = active_green_target if selected_green_target_type == "GREEN TOKEN" else None
        shared_data['navigation_target_x'] = navigation_target_x
        shared_data['navigation_target_y'] = navigation_target_y
        shared_data['navigation_mode'] = navigation_mode
        shared_data['slow_reason'] = slow_reason
        shared_data['steering_input'] = steering_input
        shared_data['acceleration_input'] = acceleration_input
        shared_data['green_target_source'] = green_target_source
        shared_data['hazard_debug'] = (
            f"HazardXY: {current_hazard_x},{current_hazard_y} | "
            f"HazErr: {None if current_hazard_error is None else round(current_hazard_error, 1)} | "
            f"Avoid: {avoidance_direction_label} | Lock: {avoidance_lock_active} | FinalSteer: {steering_input:+.2f}"
        )
        shared_data['decision_debug'] = (
            f"Reason: {selected_target_reason} | Mode: {navigation_mode} | CurLane: {current_lane} | TargetLane: {chosen_target_lane} | "
            f"CarX: {car_center_x:.1f} | GreenX: {selected_green_x} | GreenErr: {None if green_error is None else round(green_error, 1)} | "
            f"HazardX: {current_hazard_x} | HazardY: {current_hazard_y} | HazardErr: {None if current_hazard_error is None else round(current_hazard_error, 1)} | "
            f"GreenTokens: {green_token_count} | GreenSource: {green_target_source} | GreenY: {selected_green_y} | Hazards: {sorted(visible_hazard_lanes)} | "
            f"Blocking: {sorted(blocking_hazard_lanes)} | Steer: {steering_input:+.2f} | Accel: {acceleration_input:.2f} | Slow: {slow_reason}"
        )

    try:
        send_control_packet(steering_input, acceleration_input)
    except Exception as e:
        print(f"Network error: {e}")
        control_conn = None
#---------------------------------------------------------
# Main (Scheduler Initialization)
# ---------------------------------------------------------
if __name__ == '__main__':
    current_lane = START_LANE
    steering_state = 0
    tap_loop_count = 0
    last_token_lane = None
    last_token_time = 0.0
    locked_green_lane = None
    locked_green_until = 0.0
    committed_target_lane = None
    committed_target_until = 0.0
    committed_target_reason = "MAINTAIN"
    avoidance_lock_until = 0.0
    avoidance_lock_direction = 0.0
    avoidance_lock_hazard = None
    run_start_time = time.time()
    with data_lock:
        shared_data['target_lane'] = START_LANE

    print("Initializing RTSE Sample Drive...")

    # Initialize network connections
    threading.Thread(target=setup_control_server, daemon=True).start()
    threading.Thread(target=setup_cameras, daemon=True).start()
    
    print("\n--- Starting Real-Time Tasks (awaiting connections dynamically) ---\n")
    
    # Define and start real-time tasks
    t_front_camera = RTTask("ReadFrontCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_front_camera_task)
    t_back_camera = RTTask("ReadBackCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_back_camera_task)
    t_processing = RTTask("Processing", period=0.005, priority=TaskPriority.MEDIUM, execute_func=processing_task)
    t_controls = RTTask("SendControls", period=0.005, priority=TaskPriority.HIGH, execute_func=send_controls_task)
    
    # Start tasks to run concurrently
    t_front_camera.start()
    t_back_camera.start()
    t_processing.start()
    t_controls.start()

    try:
        while is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Stopping system...")
        is_running = False

    # Clean shutdown
    t_front_camera.join()
    t_back_camera.join()
    t_processing.join()
    t_controls.join()
    
    if front_camera_sock:
        front_camera_sock.close()
    if back_camera_sock:
        back_camera_sock.close()
    if control_conn:
        control_conn.close()
    cv2.destroyAllWindows()
    
    print("System terminated cleanly.")

    # Print run summary
    print("\n" + "="*45)
    print(" 🚗  END OF RUN SUMMARY  🚗")
    print("="*45)
    with data_lock:
        summary = shared_data.get('run_summary', {})
    if summary:
        print("Tokens Collected (Estimated):")
        print(f"  🟢 Green:  {summary.get('green_collected', 0)}")
        print(f"  🟡 Yellow: {summary.get('yellow_collected', 0)}")
        print(f"  🔴 Red:    {summary.get('red_collected', 0)}")
        yellow_effects = summary.get('yellow_effects', {})
        if yellow_effects:
            print("\nYellow Effects Triggered:")
            for effect_name, effect_count in yellow_effects.items():
                print(f"  - {effect_name}: {effect_count}")
        print("-" * 45)
        print("Events Detected During Run:")
        print(f"  🚓 Police Car Appeared:   {'YES' if summary.get('police_appeared') else 'NO'}")
        print(f"  🚘 Trailing Car Appeared: {'YES' if summary.get('trailing_appeared') else 'NO'}")
    print("="*45 + "\n")
