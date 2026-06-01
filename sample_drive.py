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
import os

YOLO = None

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CAMERA_HOST = '127.0.0.1'
CAMERA_READ_TIMEOUT = 0.05
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
FRONT_CAMERA_PORT = 8080
BACK_CAMERA_PORT = 8082
CONTROL_HOST = '127.0.0.1'
CONTROL_PORT = 8081
START_LANE = 0
START_CENTER_HOLD_SECONDS = 2.0
CAR_ACCELERATION = 0.84
TOKEN_SPEED_MIN_MODIFIER = 0.20
TOKEN_SPEED_MAX_MODIFIER = 1.80
MAX_ACCELERATION = TOKEN_SPEED_MAX_MODIFIER
EVENT_SPEED_PENALTY_MULTIPLIER = 0.50
GREEN_CHASE_ACCELERATION = 0.92
GREEN_STEER_GAIN = 6.0
GREEN_STEER_DEADZONE = 0.015
GREEN_MIN_STEER = 0.18
GREEN_CENTERING_Y_MIN = 120
GREEN_CENTERING_MAX_STEER = 0.35
MIN_LOOKAHEAD_Y_RATIO = 0.25
LINE_GROUP_Y_RATIO = 0.08
LANE_CHANGE_TIME_Y_RATIO = 0.14
GREEN_LOCK_SECONDS = 1.6
STEER_TAP_LOOPS = 12
STEER_RESET_LOOPS = 4
TOKEN_DECISION_Y_MIN = 40
LANE_CHANGE_TAP_LOOPS = 18
LANE_CHANGE_RESET_LOOPS = 2
TOKEN_TARGET_MEMORY_SECONDS = 2.4
GREEN_TARGET_LOCK_SECONDS = 2.8
GREEN_TARGET_LOCK_FRAMES = 18
GREEN_TARGET_MAX_MISSING_FRAMES = 5
GREEN_DECISION_MIN_Y_RATIO = 0.25
GREEN_PRE_TARGET_MAX_Y_RATIO = 0.75
GREEN_COLLECT_Y_RATIO = 0.55
GREEN_ROAD_X_MARGIN_RATIO = 0.12
GREEN_ROAD_Y_MIN_RATIO = 0.20
GREEN_ROAD_Y_MAX_RATIO = 0.92
GREEN_CENTER_DEADZONE_RATIO = 0.045
GREEN_PRE_TARGET_STEER_GAIN = 1.2
GREEN_DIRECT_STEER_GAIN = 2.5
GREEN_PREDICTION_GAIN = 0.45
YELLOW_EFFECT_DURATION_SECONDS = 5.0
TOKEN_COLLECTION_Y_RATIO = 0.90
TOKEN_COLLECTION_COOLDOWN_SECONDS = 2.0
HAZARD_AVOID_Y_MIN = 95
GREEN_BLOCKED_HAZARD_Y_MIN = 230
MIN_ACCELERATION_WHEN_SLOWED = 0.60
MIN_STEERING_ACCELERATION = 0.56
GREEN_APPROACH_STEER_DEADZONE = 0.025
GREEN_APPROACH_MIN_STEER = 0.85
GREEN_APPROACH_MIN_ACCELERATION = 0.66
GREEN_APPROACH_MAX_ACCELERATION = 0.72
GREEN_VISUAL_CORRECTION_ACCELERATION = 0.60
TOKEN_VISIBLE_MAX_ACCELERATION = 0.62
HIGH_SPEED_MAX_ACCELERATION = 0.72
HAZARD_AVOID_STEER = 1.0
HAZARD_AVOID_MAX_ACCELERATION = 0.55
TOKEN_CENTER_X_DEADZONE = 0.025
RED_COLLECTION_CENTER_X_RATIO = 0.09
TOKEN_ROW_GROUP_Y_RATIO = 0.12
HAZARD_EMERGENCY_Y_RATIO = 0.40
HAZARD_SIDE_BUFFER_Y_RATIO = 0.08
DECISION_LOCK_SECONDS = 0.25
HAZARD_DECISION_LOCK_SECONDS = 0.15
CAMERA_DELAY_SECONDS = 5.0
ACTION_DELAY_SECONDS = 5.0
YELLOW_EFFECTS = [
    'hide_next_token_type',
    'tokens_invisible',
    'camera_input_delay',
    'action_output_delay',
    'corrupted_camera_input'
]

# Yellow avoidance tunables
# Yellow tokens are treated as hazards, not bonus targets.
YELLOW_STRICT_AVOID = True
YELLOW_HAZARD_Y_MIN_RATIO = 0.18
YELLOW_CLOSE_Y_RATIO = 0.55
YELLOW_ADJACENT_BUFFER_Y_RATIO = 0.72
YELLOW_LANE_BLOCK_SCORE = 100.0

# Light detection tunables - SUPER FAST RECOVERY
LOWLIGHT_DARK_THRESHOLD = 0.55  # More sensitive detection
LOWLIGHT_RECOVERY_THRESHOLD = 0.65  # Faster recovery detection
LOWLIGHT_MIN_BASELINE = 30.0  # Lower baseline
LOWLIGHT_EVENT_WINDOW_SECONDS = 10.0
LOWLIGHT_RECOVERY_INTERVAL = 0.2  # Recovery every 0.2 seconds
LOWLIGHT_FAST_RECOVERY_COUNT = 2  # Send 2 rapid signals

# Police car detection tunables
POLICE_CONFIRM_FRAMES = 3
POLICE_SEEK_ACCELERATION = 0.70
POLICE_ROI_X_START = 0.15
POLICE_ROI_X_END = 0.85
POLICE_ROI_Y_START = 0.35
POLICE_ROI_Y_END = 0.95
POLICE_MIN_AREA = 1000
POLICE_MAX_AREA = 50000
POLICE_MIN_RED_RATIO = 0.12
POLICE_MIN_BLUE_RATIO = 0.12
POLICE_SPLIT_RATIO_MIN = 0.25
POLICE_SPLIT_RATIO_MAX = 0.75
POLICE_MIN_RED_PIXELS = 200
POLICE_MIN_BLUE_PIXELS = 200
POLICE_CONFIDENCE_THRESHOLD = 0.50
POLICE_RED_TARGET_START_SECONDS = 8.5
POLICE_EVENT_TIMEOUT_SECONDS = 10.0

# HSV ranges for police car colors
POLICE_RED_LOWER1 = np.array([0, 80, 70])
POLICE_RED_UPPER1 = np.array([10, 255, 255])
POLICE_RED_LOWER2 = np.array([170, 80, 70])
POLICE_RED_UPPER2 = np.array([180, 255, 255])
POLICE_BLUE_LOWER = np.array([100, 80, 70])
POLICE_BLUE_UPPER = np.array([140, 255, 255])

# Trailing car detection tunables
TRAILING_CONFIRM_FRAMES = 3
TRAILING_DODGE_HOLD_SECONDS = 2.0
TRAILING_MOTION_AREA_MIN = 3000
TRAILING_APPROACH_RATIO = 1.10
TRAILING_ROI_X_START = 0.25
TRAILING_ROI_X_END = 0.75
TRAILING_ROI_Y_START = 0.30
TRAILING_ROI_Y_END = 0.95
TRAILING_ASPECT_RATIO_MIN = 0.7
TRAILING_ASPECT_RATIO_MAX = 2.2
TRAILING_MIN_HEIGHT_RATIO = 0.10

# Lane-following detection tunables
LANE_FOLLOW_SMOOTHING_PREVIOUS = 0.70
LANE_FOLLOW_SMOOTHING_CURRENT = 0.30
LANE_FOLLOW_MIN_ABS_SLOPE = 0.35
LANE_FOLLOW_MAX_ABS_SLOPE = 3.5
LANE_FOLLOW_MIN_LINES_PER_SIDE = 1
LANE_FOLLOW_INTERVAL_SECONDS = 0.10
LANE_FOLLOW_PROCESS_WIDTH = 320
BACK_DETECTION_INTERVAL_SECONDS = 0.08
PROCESSING_PERIOD_SECONDS = 0.02

# Token Detection Configuration
ENABLE_TEMPLATE_MATCHING = False
ENABLE_HOUGH_CIRCLES = False
SHOW_BACK_DEBUG = False
ENABLE_HEAVY_OVERLAY = False
TOKEN_PROCESS_WIDTH = 360
TOKEN_ROI_Y_START = 0.16
TOKEN_ROI_Y_END = 0.94

YOLO_VEHICLE_CONFIDENCE = 0.50
YOLO_PROCESS_WIDTH = 416
YOLO_FOCAL_LENGTH = 1000.0
YOLO_KNOWN_CAR_WIDTH_METERS = 2.0
YOLO_VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle'}
YOLO_WEIGHTS_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'yolov8n.pt'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov8n.pt')
]

# HSV ranges for trailing car (cyan/teal)
TRAILING_CYAN_LOWER = np.array([80, 70, 70])
TRAILING_CYAN_UPPER = np.array([100, 255, 255])

# Shared Resources
shared_data = {
    'latest_front_frame': None,
    'latest_back_frame': None,
    'steering_input': 0.0,
    'acceleration_input': 0.0,
    'detected_tokens': [],
    'target_lane': START_LANE,
    'danger_detected': False,
    'decision_debug': '',
    'lights_on': True,
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
        'trailing_appeared': False,
        'light_recovery_attempts': 0,
        'light_recovered': False,
        'speed_penalties': 0
    },
    'active_yellow_effect': None,
    'yellow_effect_until': 0.0,
    'hidden_next_token_type': False,
    'last_collected_time': 0.0,
    'last_collected_line_id': None,
    'last_collected_by_lane_color': {},
    'low_light_active': False,
    'low_light_recovered': False,
    'low_light_recovery_sent': False,
    'low_light_recovery_start_time': 0.0,
    'brightness_baseline': None,
    'brightness_last': 0.0,
    'trailing_streak': 0,
    'police_streak': 0,
    'display_front_frame': None,
    'display_back_frame': None,
    'light_event_triggered': False,
    'trailing_first_triggered': False,
    'trailing_second_triggered': False,
    'police_triggered': False,
    'recovery_attempts': 0,
    'last_recovery_time': 0.0,
    'recovery_completed': False,
    'recovery_phase': 0,
    'trailing_count': 0,
    'trailing_first_time': 0.0,
    'trailing_second_time': 0.0,
    'police_active': False,
    'police_start_time': 0.0,
    'red_token_collected_during_police': False,
    'police_penalty_applied': False,
    'trailing_first_penalty_applied': False,
    'trailing_second_penalty_applied': False,
    'police_confidence': 0.0,
    'police_detection_history': [],
    'police_lane': None,
    'trailing_lane': None,
    'police_bbox': None,
    'trailing_bbox': None,
    'police_last_seen_time': 0.0,
    'trailing_last_seen_time': 0.0,
    'emergency_evasion': False,
    'evasion_target_lane': None,
    'evasion_start_time': 0.0,
    'lane_follow': {
        'lane_center_x': None,
        'left_line': None,
        'right_line': None,
        'raw_steering': 0.0,
        'smoothed_steering': 0.0,
        'valid': False
    },
    'last_lane_follow_time': 0.0,
    'last_back_detection_time': 0.0,
    'back_detection_debug': {
        'trailing_raw': False,
        'trailing_area': 0.0,
        'trailing_bbox': None,
        'trailing_lane': None,
        'police_raw': False,
        'police_confidence': 0.0,
        'police_bbox': None,
        'police_lane': None
    },
    'green_target_debug': None,
    'green_decision_debug': None,
    'police_debug': None,
    'hazard_debug': None,
    'speed_modifier': 1.0,
    'selected_target_type': 'NONE',
    'current_mode': 'LANE FOLLOW'
}
data_lock = threading.Lock()
is_running = True
front_camera_delay_buffer = []
action_delay_buffer = []

# region debug instrumentation
import os as _dbg_os
import json as _dbg_json
DEBUG_LOG_PATH = _dbg_os.path.join(_dbg_os.path.dirname(_dbg_os.path.abspath(__file__)), 'debug-a86f0d.log')
_debug_log_lock = threading.Lock()
_debug_log_times = {}

def debug_should_log(key, interval=0.5):
    now = time.time()
    with _debug_log_lock:
        last = _debug_log_times.get(key, 0.0)
        if now - last >= interval:
            _debug_log_times[key] = now
            return True
    return False

def debug_session_log(location, message, data, hypothesis_id, run_id='run1'):
    try:
        entry = {
            'sessionId': 'a86f0d',
            'runId': run_id,
            'hypothesisId': hypothesis_id,
            'location': location,
            'message': message,
            'data': data,
            'timestamp': int(time.time() * 1000),
        }
        line = _dbg_json.dumps(entry)
        with _debug_log_lock:
            with open(DEBUG_LOG_PATH, 'a', encoding='utf-8') as _f:
                _f.write(line + '\n')
    except Exception:
        pass
# endregion

# ---------------------------------------------------------
# Real-Time Scheduling Framework
# ---------------------------------------------------------
class TaskPriority:
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class RTTask(threading.Thread):
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
        except Exception:
            pass

        while is_running:
            start_time = time.time()
            self.execute_func()
            exec_time = time.time() - start_time
            sleep_time = self.period - exec_time
            
            if sleep_time > 0:
                time.sleep(sleep_time)

# ---------------------------------------------------------
# Network Connection Setup
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
# Helper Functions
# ---------------------------------------------------------

def read_single_camera(sock, window_name, data_key, display=True):
    if sock is None:
        return
        
    try:
        latest_frame_data = None
        sock.settimeout(CAMERA_READ_TIMEOUT)
        length_bytes = sock.recv(4)
        if not length_bytes:
            return
            
        image_length = int.from_bytes(length_bytes, 'little')
        if image_length <= 0 or image_length > 10_000_000:
            return
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
                
            length_bytes = sock.recv(4)
            if not length_bytes:
                return
            image_length = int.from_bytes(length_bytes, 'little')
            if image_length <= 0 or image_length > 10_000_000:
                break
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
                    frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                    cv2.imshow(window_name, frame_resized)
                
    except socket.timeout:
        return
    except Exception:
        return

def read_front_camera_task():
    read_single_camera(front_camera_sock, "Front Camera", 'latest_front_frame', display=False)

def read_back_camera_task():
    read_single_camera(back_camera_sock, "Back Camera", 'latest_back_frame', display=False)

def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i/255.0) ** invGamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def token_in_front_danger_zone(token, frame_width, frame_height):
    """
    Checks whether a token is directly in front of the player's vehicle.
    This is used as an emergency collision check for red/yellow tokens.
    """
    y = token['y']
    x = token['x']

    # Only check tokens close to the vehicle
    if y < frame_height * 0.55:
        return False

    y_ratio = y / frame_height

    # Danger zone becomes wider near the bottom of the screen
    half_width_ratio = 0.07 + 0.13 * ((y_ratio - 0.55) / 0.40)
    half_width_ratio = max(0.07, min(0.20, half_width_ratio))

    center_x = frame_width / 2.0
    return abs(x - center_x) <= frame_width * half_width_ratio

def detect_token_color_robust(frame, x, y, radius):
    x1 = max(int(x - radius), 0)
    y1 = max(int(y - radius), 0)
    x2 = min(int(x + radius), frame.shape[1])
    y2 = min(int(y + radius), frame.shape[0])
    
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "unknown"
    
    avg_color = np.mean(roi, axis=(0, 1))
    b, g, r = avg_color
    
    total = r + g + b
    if total == 0:
        return "unknown"
    
    r_ratio = r / total
    g_ratio = g / total
    b_ratio = b / total
    
    if g_ratio > 0.4 and g_ratio > r_ratio * 1.5 and g_ratio > b_ratio * 1.5:
        return "green"
    elif r_ratio > 0.45 and r_ratio > g_ratio * 1.3 and r_ratio > b_ratio * 1.3:
        return "red"
    elif r_ratio > 0.3 and g_ratio > 0.3 and r_ratio + g_ratio > 0.7:
        return "yellow"
    else:
        return "unknown"

def classify_token_roi(frame, x, y, radius):
    x1 = max(int(x - radius), 0)
    y1 = max(int(y - radius), 0)
    x2 = min(int(x + radius), frame.shape[1])
    y2 = min(int(y + radius), frame.shape[0])
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total = float(roi.shape[0] * roi.shape[1])
    green_mask = cv2.inRange(hsv, np.array((42, 45, 95), dtype=np.uint8), np.array((88, 255, 255), dtype=np.uint8))
    yellow_mask = cv2.inRange(hsv, np.array((16, 70, 105), dtype=np.uint8), np.array((38, 255, 255), dtype=np.uint8))
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array((0, 75, 95), dtype=np.uint8), np.array((12, 255, 255), dtype=np.uint8)),
        cv2.inRange(hsv, np.array((168, 75, 95), dtype=np.uint8), np.array((179, 255, 255), dtype=np.uint8))
    )

    scores = {
        "green": cv2.countNonZero(green_mask) / total,
        "yellow": cv2.countNonZero(yellow_mask) / total,
        "red": cv2.countNonZero(red_mask) / total,
    }
    color, score = max(scores.items(), key=lambda item: item[1])
    if score < 0.12:
        return "unknown"
    return color

def dedupe_token_detections(tokens):
    priority = {"green": 3, "yellow": 2, "red": 1}
    deduped = []
    for token in sorted(tokens, key=lambda t: (priority.get(t["color"], 0), t["area"]), reverse=True):
        duplicate = False
        for kept in deduped:
            dx = token["x"] - kept["x"]
            dy = token["y"] - kept["y"]
            max_dist = max(14, int(max(token["radius"], kept["radius"]) * 0.75))
            if (dx * dx + dy * dy) <= max_dist * max_dist:
                duplicate = True
                break
        if not duplicate:
            deduped.append(token)
    return deduped

def merge_token_detections(hsv_tokens):
    """
    HSV-only token detection. Template matching is disabled for performance.
    Returns HSV tokens directly with source attribution.
    """
    for token in hsv_tokens:
        token.setdefault("source", "hsv")
        token.setdefault("confidence", 1.0)
    return hsv_tokens

def detect_token_circles_by_shape(frame):
    if not ENABLE_HOUGH_CIRCLES:
        return []
    
    frame_h, frame_w = frame.shape[:2]
    roi_top = int(frame_h * 0.18)
    roi_bottom = int(frame_h * 0.90)
    roi = frame[roi_top:roi_bottom, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(18, frame_w // 18),
        param1=80,
        param2=18,
        minRadius=max(8, frame_w // 70),
        maxRadius=max(18, frame_w // 8)
    )
    if circles is None:
        return []

    tokens = []
    for x, y, radius in np.round(circles[0, :]).astype("int"):
        y += roi_top
        if y < frame_h * 0.20 or y > frame_h * 0.92:
            continue

        color = classify_token_roi(frame, x, y, radius)
        if color not in ("green", "yellow", "red"):
            continue

        tokens.append({
            "color": color,
            "x": int(x),
            "y": int(y),
            "radius": int(radius),
            "area": float(np.pi * radius * radius),
            "source": "circle"
        })
    return tokens

def detect_colored_tokens(frame):
    frame_h, frame_w = frame.shape[:2]
    roi_y1 = int(frame_h * TOKEN_ROI_Y_START)
    roi_y2 = int(frame_h * TOKEN_ROI_Y_END)
    roi_y1 = max(0, min(frame_h - 1, roi_y1))
    roi_y2 = max(roi_y1 + 1, min(frame_h, roi_y2))

    roi = frame[roi_y1:roi_y2, :]
    process_roi = roi
    scale = 1.0
    if frame_w > TOKEN_PROCESS_WIDTH:
        process_h = max(1, int(roi.shape[0] * (TOKEN_PROCESS_WIDTH / float(frame_w))))
        process_roi = cv2.resize(roi, (TOKEN_PROCESS_WIDTH, process_h), interpolation=cv2.INTER_AREA)
        scale = frame_w / float(TOKEN_PROCESS_WIDTH)

    hsv = cv2.cvtColor(process_roi, cv2.COLOR_BGR2HSV)

    detected_tokens = []
    standard_kernel = np.ones((3, 3), np.uint8)

    standard_color_ranges = {
        "green": [((42, 45, 95), (88, 255, 255))],
        "yellow": [((16, 70, 105), (38, 255, 255))],
        "red": [((0, 75, 95), (12, 255, 255)), ((168, 75, 95), (179, 255, 255))]
    }

    for color_name, ranges in standard_color_ranges.items():
        color_mask = None

        for lower, upper in ranges:
            current_mask = cv2.inRange(
                hsv,
                np.array(lower, dtype=np.uint8),
                np.array(upper, dtype=np.uint8)
            )
            color_mask = current_mask if color_mask is None else cv2.bitwise_or(color_mask, current_mask)

        _ = cv2.bitwise_and(process_roi, process_roi, mask=color_mask)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, standard_kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, standard_kernel)

        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            process_area = cv2.contourArea(cnt)
            area = process_area * scale * scale
            min_area = max(70.0, frame_w * frame_h * 0.00024)
            max_area = frame_w * frame_h * 0.20
            if area < min_area or area > max_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * process_area / (perimeter * perimeter)
            min_circularity = 0.46 if color_name in ("green", "red") else 0.58
            if circularity < min_circularity:
                continue

            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center_x = x * scale
            center_y = y * scale + roi_y1
            radius = radius * scale
            if radius <= 0:
                continue
            circle_fill_ratio = area / (np.pi * radius * radius)
            if color_name == "red" and circle_fill_ratio < 0.43:
                continue
            if center_y < frame_h * 0.20 or center_y > frame_h * 0.92:
                continue
            verified_color = classify_token_roi(frame, center_x, center_y, radius)
            if verified_color != color_name:
                continue
            x, y, width, height = cv2.boundingRect(cnt)
            if height > 0:
                aspect_ratio = width / float(height)
                if color_name == "red" and not (0.58 <= aspect_ratio <= 1.55):
                    continue
            x1 = max(0, min(frame_w - 1, int(x * scale)))
            y1 = max(0, min(frame_h - 1, int(y * scale + roi_y1)))
            x2 = max(0, min(frame_w - 1, int((x + width) * scale)))
            y2 = max(0, min(frame_h - 1, int((y + height) * scale + roi_y1)))

            detected_tokens.append({
                "color": color_name,
                "x": int(center_x),
                "y": int(center_y),
                "radius": int(radius),
                "area": float(area),
                "bbox": (x1, y1, x2, y2),
                "source": "hsv"
            })

    if ENABLE_HOUGH_CIRCLES:
        detected_tokens.extend(detect_token_circles_by_shape(frame))

    return dedupe_token_detections(detected_tokens)

def draw_light_debug_overlay(frame, tokens, current_mode):
    token_counts = {"green": 0, "yellow": 0, "red": 0}
    text_colors = {
        "green": (0, 255, 0),
        "yellow": (0, 255, 255),
        "red": (0, 0, 255),
        "hidden": (255, 255, 255)
    }
    frame_h, frame_w = frame.shape[:2]

    for token in tokens:
        color_name = token.get("color")
        if color_name in token_counts:
            token_counts[color_name] += 1
        if color_name not in ("green", "yellow", "red"):
            continue

        label = token.get("display_color", color_name)
        draw_color = text_colors.get(label, text_colors.get(color_name, (255, 255, 255)))
        center = (int(token["x"]), int(token["y"]))
        radius = int(token.get("radius", 10))
        bbox = token.get("bbox")
        if bbox:
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(frame_w - 1, int(x1)))
            y1 = max(0, min(frame_h - 1, int(y1)))
            x2 = max(0, min(frame_w - 1, int(x2)))
            y2 = max(0, min(frame_h - 1, int(y2)))
        else:
            x1 = max(0, center[0] - radius)
            y1 = max(0, center[1] - radius)
            x2 = min(frame_w - 1, center[0] + radius)
            y2 = min(frame_h - 1, center[1] + radius)

        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
        cv2.circle(frame, center, 4, draw_color, -1)
        cv2.putText(frame, label.upper(), (x1, max(16, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)

    with data_lock:
        steering = shared_data.get('steering_input', 0.0)
        acceleration = shared_data.get('acceleration_input', 0.0)
        decision_debug = shared_data.get('decision_debug', '')
        green_target_debug = shared_data.get('green_target_debug')
        green_decision_debug = shared_data.get('green_decision_debug') or {}
        police_debug = shared_data.get('police_debug') or {}
        hazard_debug = shared_data.get('hazard_debug')
        speed_modifier = shared_data.get('speed_modifier', 1.0)
        selected_target_type = shared_data.get('selected_target_type', 'NONE')
        summary = shared_data.get('run_summary', {})
        event_status = {
            'trailing': shared_data.get('trailing_detected', False),
            'police': shared_data.get('police_detected', False) or shared_data.get('police_active', False),
            'low': shared_data.get('low_light_active', False),
            'yellow_effect': shared_data.get('active_yellow_effect')
        }

    if green_target_debug:
        target_x = int(green_target_debug.get('x', frame_w // 2))
        target_y = int(green_target_debug.get('y', frame_h // 2))
        car_center = (frame_w // 2, int(frame_h * 0.92))
        target_center = (
            max(0, min(frame_w - 1, target_x)),
            max(0, min(frame_h - 1, target_y))
        )
        cv2.line(frame, car_center, target_center, (0, 255, 0), 2)
        cv2.circle(frame, target_center, 7, (0, 255, 0), 2)
        cv2.putText(frame, "GREEN TARGET", (target_center[0], max(18, target_center[1] - 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        predicted_x = int(green_target_debug.get('predicted_x', target_center[0]))
        cv2.putText(frame, f"x:{target_center[0]} y:{target_center[1]} pred:{predicted_x}", (target_center[0], max(36, target_center[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    if hazard_debug:
        for hazard in hazard_debug.get('tokens', []):
            bbox = hazard.get('bbox')
            if bbox:
                x1, y1, x2, y2 = bbox
            else:
                cx = int(hazard.get('x', frame_w // 2))
                cy = int(hazard.get('y', frame_h // 2))
                radius = int(hazard.get('radius', 18))
                x1, y1, x2, y2 = cx - radius, cy - radius, cx + radius, cy + radius
            x1 = max(0, min(frame_w - 1, int(x1)))
            y1 = max(0, min(frame_h - 1, int(y1)))
            x2 = max(0, min(frame_w - 1, int(x2)))
            y2 = max(0, min(frame_h - 1, int(y2)))
            color_name = hazard.get('color', 'red')
            danger_color = (0, 0, 255) if color_name == 'red' else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), danger_color, 3)
            cv2.putText(frame, f"AVOID {color_name.upper()}", (x1, min(frame_h - 8, y2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, danger_color, 2)

    cv2.putText(frame, f"G:{token_counts['green']} Y:{token_counts['yellow']} R:{token_counts['red']} Total:{len(tokens)}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, f"Mode: {current_mode}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Steer:{steering:.2f} Accel:{acceleration:.2f}", (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Target:{selected_target_type} Speed x{speed_modifier:.2f}", (10, 94),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    cv2.putText(frame, f"Collected G:{summary.get('green_collected', 0)} R:{summary.get('red_collected', 0)} Y:{summary.get('yellow_collected', 0)}", (10, 116),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    event_text = (
        f"Events T:{int(event_status['trailing'])} P:{int(event_status['police'])} "
        f"Low:{int(event_status['low'])} YFX:{event_status['yellow_effect'] or 'none'}"
    )
    cv2.putText(frame, event_text, (10, 138),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if decision_debug:
        cv2.putText(frame, decision_debug[:110], (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    if green_decision_debug:
        reject = green_decision_debug.get('reject', {})
        green_line = (
            f"Green cand:{green_decision_debug.get('candidate_count', 0)} "
            f"lock:{green_decision_debug.get('lock_frames', 0)} "
            f"miss:{green_decision_debug.get('missing_frames', 0)} "
            f"rej far:{reject.get('too_far', 0)} roi:{reject.get('outside_roi', 0)} "
            f"haz:{reject.get('blocked_by_hazard', 0)} evt:{reject.get('event_override', 0)} nolock:{reject.get('no_lock', 0)}"
        )
        cv2.putText(frame, green_line[:130], (10, 182),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1)
    if police_debug:
        police_line = (
            f"Police active:{int(police_debug.get('active', False))} "
            f"det:{int(police_debug.get('detected', False))} "
            f"t:{police_debug.get('elapsed', 0.0):.1f}s "
            f"left:{police_debug.get('remaining', 0.0):.1f}s "
            f"red_req:{int(police_debug.get('red_required', False))} "
            f"{police_debug.get('reason', '')}"
        )
        cv2.putText(frame, police_line[:130], (10, 204),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1)
    return frame

def draw_detected_tokens(frame, tokens):
    display_frame = frame.copy()
    frame_h, frame_w = display_frame.shape[:2]

    text_colors = {
        "green": (0, 255, 0),
        "yellow": (0, 255, 255),
        "red": (0, 0, 255),
        "hidden": (255, 255, 255)
    }

    overlay = display_frame.copy()
    for token in tokens:
        label = token.get("display_color", token["color"])
        source = token.get("source")
        draw_label = label.upper() if not source else f"{label.upper()}:{source[0]}"
        color = text_colors.get(label, text_colors.get(token["color"], (255, 255, 255)))
        center = (token["x"], token["y"])
        radius = token["radius"]
        cv2.circle(overlay, center, radius, color, -1)
        bbox = token.get("bbox")
        if bbox:
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(frame_w - 1, int(x1)))
            y1 = max(0, min(frame_h - 1, int(y1)))
            x2 = max(0, min(frame_w - 1, int(x2)))
            y2 = max(0, min(frame_h - 1, int(y2)))
        else:
            x1 = max(center[0]-radius, 0)
            y1 = max(center[1]-radius, 0)
            x2 = min(center[0]+radius, frame_w-1)
            y2 = min(center[1]+radius, frame_h-1)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(display_frame, center, 4, color, -1)
        cv2.putText(display_frame, draw_label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    alpha = 0.3
    cv2.addWeighted(overlay, alpha, display_frame, 1-alpha, 0, display_frame)
    return display_frame

<<<<<<< HEAD
def draw_vehicle_box(display_frame, bbox, source_shape, label, lane=None, confidence=None, color=(255, 255, 255)):
    """
    Draw a scaled vehicle bounding box on the display frame.
    bbox must come from the original camera frame as (x, y, w, h).
    """
    if bbox is None or source_shape is None:
        return display_frame

    src_h, src_w = source_shape[:2]
    dst_h, dst_w = display_frame.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return display_frame

    scale_x = dst_w / float(src_w)
    scale_y = dst_h / float(src_h)

    x, y, w, h = bbox
    x1 = int(max(0, min(dst_w - 1, x * scale_x)))
    y1 = int(max(0, min(dst_h - 1, y * scale_y)))
    x2 = int(max(0, min(dst_w - 1, (x + w) * scale_x)))
    y2 = int(max(0, min(dst_h - 1, (y + h) * scale_y)))

    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)

    label_parts = [label]
    if lane is not None:
        label_parts.append(f"Lane {lane}")
    if confidence is not None:
        label_parts.append(f"{confidence:.2f}")
    text = " | ".join(label_parts)

    text_y = y1 - 10 if y1 > 25 else y2 + 25
    cv2.putText(display_frame, text, (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return display_frame
=======
def make_lane_boundary(line_params, y_bottom, y_top):
    slope, intercept = line_params
    if abs(slope) < 0.001:
        return None

    x_bottom = int((y_bottom - intercept) / slope)
    x_top = int((y_top - intercept) / slope)
    return (x_bottom, y_bottom, x_top, y_top)

def detect_lane_following(frame, previous_steering=0.0):
    """
    Detect road lane boundaries using Canny + polygon ROI + HoughLinesP.
    Returns lane-center and steering data without touching token detection.
    """
    if frame is None:
        return {
            'lane_center_x': None,
            'left_line': None,
            'right_line': None,
            'raw_steering': 0.0,
            'smoothed_steering': previous_steering,
            'valid': False
        }

    original_height, original_width = frame.shape[:2]
    process_frame = frame
    scale_x = 1.0
    scale_y = 1.0
    if original_width > LANE_FOLLOW_PROCESS_WIDTH:
        process_height = int(original_height * (LANE_FOLLOW_PROCESS_WIDTH / float(original_width)))
        process_frame = cv2.resize(frame, (LANE_FOLLOW_PROCESS_WIDTH, process_height))
        scale_x = original_width / float(LANE_FOLLOW_PROCESS_WIDTH)
        scale_y = original_height / float(process_height)

    frame_height, frame_width = process_frame.shape[:2]
    gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)

    height, width = edges.shape
    polygon = np.array([
        [
            (0, height),
            (int(width * 0.2), int(height * 0.6)),
            (int(width * 0.8), int(height * 0.6)),
            (width, height)
        ]
    ], np.int32)

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, polygon, 255)
    roi = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        roi,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )

    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue

            slope = dy / float(dx)
            if abs(slope) < LANE_FOLLOW_MIN_ABS_SLOPE or abs(slope) > LANE_FOLLOW_MAX_ABS_SLOPE:
                continue

            intercept = y1 - slope * x1
            if slope < 0:
                left_lines.append((slope, intercept))
            else:
                right_lines.append((slope, intercept))

    y_bottom = frame_height - 1
    y_top = int(frame_height * 0.62)
    left_line = None
    right_line = None

    if len(left_lines) >= LANE_FOLLOW_MIN_LINES_PER_SIDE:
        left_line = make_lane_boundary(np.mean(left_lines, axis=0), y_bottom, y_top)
    if len(right_lines) >= LANE_FOLLOW_MIN_LINES_PER_SIDE:
        right_line = make_lane_boundary(np.mean(right_lines, axis=0), y_bottom, y_top)

    lane_center_x = None
    raw_steering = 0.0
    smoothed_steering = previous_steering
    valid = False

    if left_line is not None and right_line is not None:
        left_x = left_line[0]
        right_x = right_line[0]
        if left_x < right_x:
            lane_center_x = (left_x + right_x) / 2.0
            image_center_x = frame_width / 2.0
            error = lane_center_x - image_center_x
            raw_steering = error / image_center_x
            raw_steering = max(-1.0, min(1.0, raw_steering))
            smoothed_steering = (
                previous_steering * LANE_FOLLOW_SMOOTHING_PREVIOUS +
                raw_steering * LANE_FOLLOW_SMOOTHING_CURRENT
            )
            smoothed_steering = max(-1.0, min(1.0, smoothed_steering))
            valid = True

    if left_line is not None:
        left_line = (
            int(left_line[0] * scale_x),
            int(left_line[1] * scale_y),
            int(left_line[2] * scale_x),
            int(left_line[3] * scale_y)
        )
    if right_line is not None:
        right_line = (
            int(right_line[0] * scale_x),
            int(right_line[1] * scale_y),
            int(right_line[2] * scale_x),
            int(right_line[3] * scale_y)
        )
    if lane_center_x is not None:
        lane_center_x *= scale_x

    return {
        'lane_center_x': lane_center_x,
        'left_line': left_line,
        'right_line': right_line,
        'raw_steering': raw_steering,
        'smoothed_steering': smoothed_steering,
        'valid': valid
    }

def draw_lane_following_overlay(frame, lane_info, current_mode):
    if frame is None:
        return frame

    frame_h, frame_w = frame.shape[:2]
    lane_info = lane_info or {}
    left_line = lane_info.get('left_line')
    right_line = lane_info.get('right_line')
    lane_center_x = lane_info.get('lane_center_x')
    steering = lane_info.get('smoothed_steering', 0.0)

    if left_line is not None:
        cv2.line(frame, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (255, 0, 0), 3)
        cv2.putText(frame, "Left lane", (max(5, left_line[2] - 90), max(20, left_line[3] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)

    if right_line is not None:
        cv2.line(frame, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 165, 255), 3)
        cv2.putText(frame, "Right lane", (min(frame_w - 115, right_line[2] + 8), max(20, right_line[3] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2)

    image_center_x = int(frame_w / 2)
    cv2.line(frame, (image_center_x, int(frame_h * 0.62)), (image_center_x, frame_h - 1), (255, 255, 255), 1)

    if lane_center_x is not None:
        center_pt = (int(lane_center_x), int(frame_h * 0.88))
        cv2.circle(frame, center_pt, 7, (0, 255, 255), -1)
        cv2.line(frame, (int(lane_center_x), int(frame_h * 0.62)), (int(lane_center_x), frame_h - 1), (0, 255, 255), 2)
        lane_center_text = f"Lane center X: {lane_center_x:.1f}"
    else:
        lane_center_text = "Lane center X: None"

    cv2.putText(frame, f"Mode: {current_mode}", (10, frame_h - 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, lane_center_text, (10, frame_h - 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, f"Steering: {steering:.2f}", (10, frame_h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame

def estimate_vehicle_distance(bbox_width, bbox_height):
    if bbox_width <= 0:
        return None
    return (YOLO_KNOWN_CAR_WIDTH_METERS * YOLO_FOCAL_LENGTH) / float(bbox_width)

yolo_vehicle_model = None
yolo_vehicle_model_checked = False
yolo_vehicle_model_available = False

def get_yolo_vehicle_model():
    global yolo_vehicle_model, yolo_vehicle_model_checked, yolo_vehicle_model_available

    if yolo_vehicle_model_checked:
        return yolo_vehicle_model

    yolo_vehicle_model_checked = True
    if YOLO is None:
        print("[YOLO] ultralytics is not installed; using color-based car detection fallback.")
        return None

    weights_path = None
    for candidate in YOLO_WEIGHTS_CANDIDATES:
        if os.path.exists(candidate):
            weights_path = candidate
            break

    if weights_path is None:
        print("[YOLO] weights/yolov8n.pt not found; using color-based car detection fallback.")
        return None

    try:
        yolo_vehicle_model = YOLO(weights_path)
        yolo_vehicle_model_available = True
        print(f"[YOLO] Loaded vehicle detector: {weights_path}")
    except Exception as e:
        print(f"[YOLO] Could not load YOLOv8 model: {e}")
        yolo_vehicle_model = None
        yolo_vehicle_model_available = False

    return yolo_vehicle_model

def detect_yolo_vehicles(frame):
    model = get_yolo_vehicle_model()
    if model is None or frame is None:
        return []

    frame_h, frame_w = frame.shape[:2]
    process_frame = frame
    scale_x = 1.0
    scale_y = 1.0
    if frame_w > YOLO_PROCESS_WIDTH:
        process_h = int(frame_h * (YOLO_PROCESS_WIDTH / float(frame_w)))
        process_frame = cv2.resize(frame, (YOLO_PROCESS_WIDTH, process_h))
        scale_x = frame_w / float(YOLO_PROCESS_WIDTH)
        scale_y = frame_h / float(process_h)

    detections = []
    try:
        results = model(process_frame, verbose=False)
    except Exception as e:
        if debug_should_log("yolo_error", interval=2.0):
            print(f"[YOLO] Detection error: {e}")
        return detections

    for result in results:
        boxes = getattr(result, 'boxes', None)
        if boxes is None:
            continue

        for box in boxes:
            cls = int(box.cls[0])
            name = model.names.get(cls, str(cls))
            conf = float(box.conf[0])
            if name not in YOLO_VEHICLE_CLASSES or conf < YOLO_VEHICLE_CONFIDENCE:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            bbox_w = max(1, x2 - x1)
            bbox_h = max(1, y2 - y1)
            center_x = x1 + bbox_w / 2.0
            center_y = y1 + bbox_h / 2.0

            detections.append({
                'class_name': name,
                'confidence': conf,
                'bbox': (x1, y1, bbox_w, bbox_h),
                'lane': lane_from_x(center_x, center_y, frame_width=frame_w, frame_height=frame_h),
                'distance': estimate_vehicle_distance(bbox_w, bbox_h),
                'area': float(bbox_w * bbox_h)
            })

    return detections

def detect_trailing_car_yolo(back_frame, vehicles):
    if not vehicles:
        return False, 0.0, None, None

    frame_h, frame_w = back_frame.shape[:2]
    roi_x1 = int(frame_w * TRAILING_ROI_X_START)
    roi_x2 = int(frame_w * TRAILING_ROI_X_END)
    roi_y1 = int(frame_h * TRAILING_ROI_Y_START)
    roi_y2 = int(frame_h * TRAILING_ROI_Y_END)

    candidates = []
    for vehicle in vehicles:
        x, y, w, h = vehicle['bbox']
        center_x = x + w / 2.0
        center_y = y + h / 2.0
        if not (roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2):
            continue
        height_ratio = h / float(frame_h)
        if height_ratio < TRAILING_MIN_HEIGHT_RATIO:
            continue
        candidates.append(vehicle)

    if not candidates:
        return False, 0.0, None, None

    closest = max(candidates, key=lambda item: item['area'])
    bbox = closest['bbox']
    lane = closest['lane']
    area = closest['area']
    trailing = area >= TRAILING_MOTION_AREA_MIN
    return trailing, area, bbox, lane

def detect_police_car_yolo(back_frame, vehicles):
    if not vehicles:
        return False, 0.0, None, None

    best_confidence = 0.0
    best_bbox = None
    best_lane = None
    for vehicle in vehicles:
        x, y, w, h = vehicle['bbox']
        roi = back_frame[max(0, y):min(back_frame.shape[0], y + h), max(0, x):min(back_frame.shape[1], x + w)]
        if roi.size == 0:
            continue

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv_roi, POLICE_RED_LOWER1, POLICE_RED_UPPER1),
            cv2.inRange(hsv_roi, POLICE_RED_LOWER2, POLICE_RED_UPPER2)
        )
        blue_mask = cv2.inRange(hsv_roi, POLICE_BLUE_LOWER, POLICE_BLUE_UPPER)
        total_pixels = float(roi.shape[0] * roi.shape[1])
        red_ratio = cv2.countNonZero(red_mask) / total_pixels
        blue_ratio = cv2.countNonZero(blue_mask) / total_pixels

        if red_ratio < POLICE_MIN_RED_RATIO or blue_ratio < POLICE_MIN_BLUE_RATIO:
            continue

        color_confidence = min(1.0, (red_ratio / POLICE_MIN_RED_RATIO + blue_ratio / POLICE_MIN_BLUE_RATIO) / 2.0)
        confidence = min(1.0, vehicle['confidence'] * 0.6 + color_confidence * 0.4)
        if confidence > best_confidence:
            best_confidence = confidence
            best_bbox = vehicle['bbox']
            best_lane = vehicle['lane']

    return best_confidence >= POLICE_CONFIDENCE_THRESHOLD, best_confidence, best_bbox, best_lane

def draw_yolo_vehicle_overlay(frame, vehicles):
    for vehicle in vehicles:
        x, y, w, h = vehicle['bbox']
        distance = vehicle.get('distance')
        label = f"{vehicle['class_name']} {vehicle['confidence']:.2f}"
        if distance is not None:
            label += f" {distance:.1f}m"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, label, (x, max(18, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return frame
>>>>>>> 17af060b8a416cdfc3d1217c5a395dad26e52bbf

def lane_from_x(x, y=None, frame_width=640, frame_height=480):
    if y is None:
        lane_count = 5
        lane_width = frame_width / lane_count
        lane = int(x // lane_width) - 2
        return max(-2, min(2, lane))

    if y <= frame_height * 0.62:
        lane_count = 5
        lane_width = frame_width / lane_count
        lane = int(x // lane_width) - 2
        return max(-2, min(2, lane))
        
    vp_x = frame_width / 2.0
    vp_y = frame_height * 0.45
    
    if y <= vp_y:
        lane_count = 5
        lane_width = frame_width / lane_count
        lane = int(x // lane_width) - 2
        return max(-2, min(2, lane))
        
    dy = y - vp_y
    dx = x - vp_x
    slope = dx / dy
    
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

def detect_trailing_car_enhanced(back_frame):
    """
    ENHANCED trailing car detection with cyan/teal color detection
    """
    if back_frame is None:
        return False, 0.0, None, None

    frame_h, frame_w = back_frame.shape[:2]

    roi_x1 = int(frame_w * TRAILING_ROI_X_START)
    roi_x2 = int(frame_w * TRAILING_ROI_X_END)
    roi_y1 = int(frame_h * TRAILING_ROI_Y_START)
    roi_y2 = int(frame_h * TRAILING_ROI_Y_END)

    roi = back_frame[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_h, roi_w = roi.shape[:2]
    
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    cyan_mask = cv2.inRange(hsv_roi, TRAILING_CYAN_LOWER, TRAILING_CYAN_UPPER)
    cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    contours, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    trailing = False
    largest_area = 0.0
    largest_bbox = None
    trailing_lane = None
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < TRAILING_MOTION_AREA_MIN:
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h > 0 else 0.0
        height_ratio = float(h) / roi_h
        
        if (TRAILING_ASPECT_RATIO_MIN <= aspect_ratio <= TRAILING_ASPECT_RATIO_MAX and
            height_ratio > TRAILING_MIN_HEIGHT_RATIO and
            height_ratio < 0.8):
            
            if area > largest_area:
                largest_area = area
                # Get lane of trailing car
                center_x = x + w/2
                trailing_lane = lane_from_x(center_x + roi_x1, frame_height=frame_h, frame_width=frame_w)
                largest_bbox = (x + roi_x1, y + roi_y1, w, h)
                
                with data_lock:
                    prev_area = shared_data.get('back_prev_area', 0.0)
                    if prev_area > 0 and area > (prev_area * TRAILING_APPROACH_RATIO):
                        trailing = True
                        print(f"[TRAILING] Cyan car detected in lane {trailing_lane}! Area: {area:.0f}")

    with data_lock:
        if largest_area > 0:
            shared_data['back_prev_area'] = largest_area

    return trailing, largest_area, largest_bbox, trailing_lane

def detect_police_car_accurate(back_frame):
    """
    ACCURATE police car detection using split red/blue color detection
    """
    if back_frame is None:
        return False, 0.0, None, None

    frame_h, frame_w = back_frame.shape[:2]

    roi_x1 = int(frame_w * POLICE_ROI_X_START)
    roi_x2 = int(frame_w * POLICE_ROI_X_END)
    roi_y1 = int(frame_h * POLICE_ROI_Y_START)
    roi_y2 = int(frame_h * POLICE_ROI_Y_END)

    roi = back_frame[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_h, roi_w = roi.shape[:2]
    
    if roi_h == 0 or roi_w == 0:
        return False, 0.0, None, None
    
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    red_mask1 = cv2.inRange(hsv_roi, POLICE_RED_LOWER1, POLICE_RED_UPPER1)
    red_mask2 = cv2.inRange(hsv_roi, POLICE_RED_LOWER2, POLICE_RED_UPPER2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    blue_mask = cv2.inRange(hsv_roi, POLICE_BLUE_LOWER, POLICE_BLUE_UPPER)
    
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_regions = []
    for cnt in red_contours:
        area = cv2.contourArea(cnt)
        if area > 80:
            x, y, w, h = cv2.boundingRect(cnt)
            red_regions.append((x, y, w, h, area))
    
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_regions = []
    for cnt in blue_contours:
        area = cv2.contourArea(cnt)
        if area > 80:
            x, y, w, h = cv2.boundingRect(cnt)
            blue_regions.append((x, y, w, h, area))
    
    police = False
    best_confidence = 0.0
    best_bbox = None
    police_lane = None
    
    if red_regions and blue_regions:
        for rx, ry, rw, rh, rarea in red_regions:
            for bx, by, bw, bh, barea in blue_regions:
                x1 = min(rx, bx)
                y1 = min(ry, by)
                x2 = max(rx + rw, bx + bw)
                y2 = max(ry + rh, by + bh)
                
                combined_w = x2 - x1
                combined_h = y2 - y1
                combined_area = combined_w * combined_h
                
                if combined_area < POLICE_MIN_AREA or combined_area > POLICE_MAX_AREA:
                    continue
                
                aspect_ratio = float(combined_w) / combined_h if combined_h > 0 else 0
                if aspect_ratio < 0.5 or aspect_ratio > 2.5:
                    continue
                
                combined_roi = roi[y1:y2, x1:x2]
                if combined_roi.size == 0:
                    continue
                
                hsv_combined = cv2.cvtColor(combined_roi, cv2.COLOR_BGR2HSV)
                red_combined = cv2.bitwise_or(
                    cv2.inRange(hsv_combined, POLICE_RED_LOWER1, POLICE_RED_UPPER1),
                    cv2.inRange(hsv_combined, POLICE_RED_LOWER2, POLICE_RED_UPPER2)
                )
                blue_combined = cv2.inRange(hsv_combined, POLICE_BLUE_LOWER, POLICE_BLUE_UPPER)
                
                red_pixels = cv2.countNonZero(red_combined)
                blue_pixels = cv2.countNonZero(blue_combined)
                total_pixels = combined_roi.shape[0] * combined_roi.shape[1]
                
                if total_pixels == 0:
                    continue
                
                red_ratio = red_pixels / total_pixels
                blue_ratio = blue_pixels / total_pixels
                
                if red_ratio < POLICE_MIN_RED_RATIO or blue_ratio < POLICE_MIN_BLUE_RATIO:
                    continue
                
                split_ratio = red_pixels / (red_pixels + blue_pixels) if (red_pixels + blue_pixels) > 0 else 0.5
                
                if POLICE_SPLIT_RATIO_MIN <= split_ratio <= POLICE_SPLIT_RATIO_MAX:
                    confidence = 0.0
                    
                    color_presence = min(1.0, (red_pixels / POLICE_MIN_RED_PIXELS + blue_pixels / POLICE_MIN_BLUE_PIXELS) / 2.0)
                    confidence += color_presence * 0.4
                    
                    balance = 1.0 - abs(split_ratio - 0.5) * 2.0
                    confidence += balance * 0.3
                    
                    area_factor = min(1.0, combined_area / 8000.0)
                    confidence += area_factor * 0.3
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        # Get lane of police car
                        center_x = x1 + combined_w/2
                        police_lane = lane_from_x(center_x + roi_x1, frame_height=frame_h, frame_width=frame_w)
                        best_bbox = (x1 + roi_x1, y1 + roi_y1, combined_w, combined_h)
    
    if best_confidence >= POLICE_CONFIDENCE_THRESHOLD:
        police = True
        print(f"[POLICE] Police car detected in lane {police_lane}! Confidence: {best_confidence:.2f}")
    
    return police, best_confidence, best_bbox, police_lane

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

def send_fast_recovery_burst():
    """
    Send a fast recovery burst with proper timing to avoid freezing
    """
    global control_conn
    
    if control_conn is None:
        return False
    
    try:
        for i in range(LOWLIGHT_FAST_RECOVERY_COUNT):
            data = struct.pack('ff', 0.0, -1.0)
            control_conn.sendall(data)
            time.sleep(0.05)
        return True
    except Exception as e:
        print(f"[LIGHT] Failed to send fast recovery: {e}")
        return False

def get_lane_center_x(lane, frame_width):
    """Get the center x coordinate of a lane"""
    lane_width = frame_width / 5.0
    return (lane + 2) * lane_width + lane_width / 2.0

def get_safe_lane(current_lane, hazard_lanes, frame_width, bonus_lanes=None):
    """Find the safest lane to move to avoiding hazards"""
    if bonus_lanes is None:
        bonus_lanes = set()
        
    available_lanes = [l for l in range(-2, 3) if l not in hazard_lanes]
    if not available_lanes:
        return current_lane
    
    best_lane = current_lane
    best_score = -1
    
    for lane in available_lanes:
        dist = abs(lane - current_lane)
        score = 100.0 - (dist * 18.0)
        
        if lane in bonus_lanes:
            score += 25.0
        if lane == current_lane:
            score += 8.0
        if abs(lane) == 2:
            score -= 4.0
        
        if score > best_score:
            best_score = score
            best_lane = lane
            
    return best_lane

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))

def green_rejection_reason(token, frame_width, frame_height, current_lane, blocked_lanes):
    if token.get('color') != 'green':
        return "not_green"
    x = token.get('x', 0)
    y = token.get('y', 0)
    if y < frame_height * GREEN_DECISION_MIN_Y_RATIO:
        return "too_far"
    if y > frame_height * GREEN_ROAD_Y_MAX_RATIO:
        return "outside_roi"
    if x < frame_width * GREEN_ROAD_X_MARGIN_RATIO or x > frame_width * (1.0 - GREEN_ROAD_X_MARGIN_RATIO):
        return "outside_roi"

    token_lane = lane_from_x(x, y, frame_width=frame_width, frame_height=frame_height)
    if token_lane in blocked_lanes:
        return "blocked_by_hazard"
    if abs(token_lane - current_lane) > 1 and y < frame_height * GREEN_PRE_TARGET_MAX_Y_RATIO:
        return "unreachable_lane"
    return None

def is_reachable_green_token(token, frame_width, frame_height, current_lane, blocked_lanes):
    return green_rejection_reason(token, frame_width, frame_height, current_lane, blocked_lanes) is None

def green_target_score(token, frame_width, frame_height, current_lane):
    token_lane = lane_from_x(token['x'], token['y'], frame_width=frame_width)
    center_error = abs(token['x'] - (frame_width / 2.0)) / (frame_width / 2.0)
    lane_center_error = abs(token['x'] - get_lane_center_x(token_lane, frame_width)) / (frame_width / 5.0)
    lane_change_cost = abs(token_lane - current_lane) * 0.08
    y_ratio = token['y'] / float(frame_height)
    future_bonus = 80.0 if 0.25 <= y_ratio <= 0.75 else 0.0
    return (
        token['y'] * 2.0
        + future_bonus
        - center_error * 120.0
        - lane_center_error * 60.0
        - lane_change_cost * frame_width
    )

def select_green_target(green_tokens, previous_target, frame_width, frame_height, current_lane, blocked_lanes):
    reachable = [
        token for token in green_tokens
        if is_reachable_green_token(token, frame_width, frame_height, current_lane, blocked_lanes)
    ]
    if not reachable:
        return None, []

    if previous_target:
        previous_x = previous_target.get('x')
        previous_y = previous_target.get('y')
        if previous_x is not None and previous_y is not None:
            lock_radius = max(45.0, frame_width * 0.11)
            locked_candidates = []
            for token in reachable:
                dx = token['x'] - previous_x
                dy = token['y'] - previous_y
                if (dx * dx + dy * dy) <= lock_radius * lock_radius:
                    locked_candidates.append(token)
            if locked_candidates:
                return max(locked_candidates, key=lambda t: green_target_score(t, frame_width, frame_height, current_lane)), reachable

    return max(reachable, key=lambda t: green_target_score(t, frame_width, frame_height, current_lane)), reachable

def maybe_record_collected_token(tokens, frame_width, frame_height, current_lane, front_frame):
    global locked_green_lane, locked_green_until, last_token_lane, last_token_time, committed_target_lane, committed_target_until, committed_target_reason, locked_green_target, locked_green_target_frames, locked_green_target_missing_frames

    now = time.time()
    collection_y = frame_height * TOKEN_COLLECTION_Y_RATIO
    collected = None

    for token in tokens:
        if token['y'] < collection_y:
            continue

        token_lane = lane_from_x(token['x'], token['y'], frame_width=frame_width, frame_height=frame_height)
        token_is_centered = abs(token['x'] - (frame_width / 2.0)) <= frame_width * 0.16
        if token_lane != current_lane and not token_is_centered:
            continue

<<<<<<< HEAD
        # Avoid yellow tokens completely. Red is only allowed during police-active mode.
        true_color = token.get('true_color', token.get('color'))

        if true_color not in ['green', 'red', 'yellow']:
            true_color = detect_token_color_robust(front_frame, token['x'], token['y'], token['radius'])

        with data_lock:
            police_active_now = shared_data.get('police_active', False)

        if true_color == 'yellow':
            continue

        if true_color == 'red' and not police_active_now:
            continue

        if true_color not in ['green', 'red']:
=======
        true_color = token.get('true_color', token.get('color'))
        if true_color not in ['green', 'yellow', 'red']:
            true_color = detect_token_color_robust(front_frame, token['x'], token['y'], token['radius'])
        
        if true_color not in ['green', 'yellow', 'red']:
>>>>>>> 17af060b8a416cdfc3d1217c5a395dad26e52bbf
            continue

        token_id = f"{token_lane}:{true_color}"

        with data_lock:
            recent_collections = shared_data.get('last_collected_by_lane_color', {})
            recent_same_token = (
                now - recent_collections.get(token_id, 0.0) < TOKEN_COLLECTION_COOLDOWN_SECONDS
            )

        if recent_same_token:
            continue

        collected = (token_id, true_color)
        break

    if collected is None:
        return

    token_id, color = collected
    with data_lock:
        shared_data['last_collected_line_id'] = token_id
        shared_data['last_collected_time'] = now
        shared_data['last_collected_by_lane_color'][token_id] = now
        shared_data['run_summary'][f'{color}_collected'] += 1
        
        if color == 'red' and shared_data.get('police_active', False):
            shared_data['red_token_collected_during_police'] = True
            print(f"[POLICE] Red token collected! Police threat neutralized.")

    if color == 'yellow':
        start_random_yellow_effect()
    elif color == 'green':
        locked_green_lane = None
        locked_green_until = 0.0
        locked_green_target = None
        locked_green_target_frames = 0
        locked_green_target_missing_frames = 0
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

def calculate_speed_modifier(green_count, red_count, penalty_count):
    token_modifier = 1.0 + (green_count * 0.10) - (red_count * 0.20)
    token_modifier = max(TOKEN_SPEED_MIN_MODIFIER, min(TOKEN_SPEED_MAX_MODIFIER, token_modifier))
    return token_modifier * (EVENT_SPEED_PENALTY_MULTIPLIER ** penalty_count)

def apply_tap_steering(desired_lane):
    global current_lane, steering_state, tap_loop_count, committed_target_lane

    desired_lane = max(-2, min(2, desired_lane))

    if steering_state == 0 and desired_lane != current_lane:
        step = 1 if desired_lane > current_lane else -1
        committed_target_lane = current_lane + step
        steering_state = 1
        tap_loop_count = 0

    if steering_state == 1:
        tap_loop_count += 1
        if tap_loop_count >= 1:
            steering_state = 2
            tap_loop_count = 0
        return 0.0

    if steering_state == 2:
        direction = 1.0 if committed_target_lane > current_lane else -1.0
        tap_loop_count += 1
        if tap_loop_count >= LANE_CHANGE_TAP_LOOPS:
            current_lane = committed_target_lane
            steering_state = 3
            tap_loop_count = 0
        return direction

    if steering_state == 3:
        tap_loop_count += 1
        if tap_loop_count >= LANE_CHANGE_RESET_LOOPS:
            steering_state = 0
            tap_loop_count = 0
            committed_target_lane = None
        return 0.0

    steering_state = 0
    tap_loop_count = 0
    committed_target_lane = None
    return 0.0

def make_camera_placeholder(title, status):
    frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
    cv2.putText(frame, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, status, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
    cv2.putText(frame, "Start SpeedTrials2D.exe after this script", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    return frame

def update_camera_windows():
    with data_lock:
        front_disp = shared_data.get('display_front_frame')
        back_disp = shared_data.get('display_back_frame')
    if front_disp is None:
        front_disp = make_camera_placeholder("Front Camera", "Waiting for controller...")
    if back_disp is None:
        back_disp = make_camera_placeholder("Back Camera", "Waiting for controller...")
    cv2.imshow("Front Camera", front_disp)
    cv2.imshow("Back Camera", back_disp)
    cv2.waitKey(1)

# Global variables for steering state machine
steering_state = 0
tap_loop_count = 0
current_lane = START_LANE
last_token_lane = None
last_token_time = 0.0
locked_green_lane = None
locked_green_until = 0.0
committed_target_lane = None
committed_target_until = 0.0
committed_target_reason = "MAINTAIN"
last_token_debug_time = 0.0
run_start_time = None
trailing_dodge_lane = None
trailing_dodge_until = 0.0
last_trailing_notice = None
last_police_notice = None
locked_green_target = None
locked_green_target_frames = 0
locked_green_target_missing_frames = 0

def processing_task():
    global current_lane, steering_state, tap_loop_count, last_token_lane, last_token_time, locked_green_lane, locked_green_until, committed_target_lane, committed_target_until, committed_target_reason, last_token_debug_time, run_start_time, trailing_dodge_lane, trailing_dodge_until, last_trailing_notice, last_police_notice
    
    with data_lock:
        front_frame = shared_data['latest_front_frame']
        back_frame = shared_data.get('latest_back_frame')
        front_connected = front_camera_sock is not None
        back_connected = back_camera_sock is not None

    if front_frame is not None:
        try:
            processed_front_frame = apply_camera_effects(front_frame)
            # HSV is the primary and only token detector - fast and lightweight
            hsv_tokens = detect_colored_tokens(processed_front_frame)
            
            tokens = merge_token_detections(hsv_tokens)
            
            # SUPER FAST LIGHT DETECTION - Check brightness every frame
            try:
                gray_full = cv2.cvtColor(front_frame, cv2.COLOR_BGR2GRAY)
                current_brightness = float(np.mean(gray_full))
                
                with data_lock:
                    baseline = shared_data.get('brightness_baseline')
                    low_light_active = shared_data.get('low_light_active', False)
                    
                    if baseline is None:
                        baseline = current_brightness
                        shared_data['brightness_baseline'] = baseline
                        print(f"[LIGHT] Baseline brightness set to: {baseline:.1f}")
                    
                    # Check if camera is split (half black) - NOT a low light condition
                    # Check standard deviation of brightness to detect split screen
                    std_dev = np.std(gray_full)
                    is_split_screen = std_dev > 80  # High variance means split screen
                    
                    # Only detect low light if NOT split screen and brightness is low
                    is_dark = False
                    if not is_split_screen and baseline >= LOWLIGHT_MIN_BASELINE:
                        if current_brightness < baseline * LOWLIGHT_DARK_THRESHOLD:
                            is_dark = True
                            if not low_light_active:
                                print(f"[LIGHT] ⚡ DARKNESS DETECTED! Brightness: {current_brightness:.1f}")
                                shared_data['recovery_phase'] = 1
                        elif low_light_active and current_brightness < baseline * LOWLIGHT_RECOVERY_THRESHOLD:
                            is_dark = True
                    
                    # Update state
                    if is_dark and not low_light_active:
                        shared_data['low_light_active'] = True
                        shared_data['low_light_recovery_sent'] = False
                        shared_data['low_light_recovery_start_time'] = time.time()
                        shared_data['recovery_attempts'] = 0
                        shared_data['recovery_completed'] = False
                        print(f"[LIGHT] 🔦 LOW LIGHT ACTIVE! Sending fast recovery...")
                    elif not is_dark and low_light_active:
                        shared_data['low_light_active'] = False
                        shared_data['low_light_recovered'] = True
                        shared_data['lights_on'] = True
                        shared_data['low_light_recovery_sent'] = False
                        shared_data['recovery_completed'] = True
                        shared_data['recovery_phase'] = 0
                        shared_data['run_summary']['light_recovered'] = True
                        print(f"[LIGHT] ✅ LIGHT RESTORED! Brightness: {current_brightness:.1f}")
                    
                    # Update baseline: fast when normal, very slow when in dark mode to recover from false positives
                    if not low_light_active:
                        baseline = baseline * 0.97 + current_brightness * 0.03
                    else:
                        baseline = baseline * 0.99 + current_brightness * 0.01
                    shared_data['brightness_baseline'] = baseline
                    
                    shared_data['brightness_last'] = current_brightness
                    shared_data['lights_on'] = not low_light_active
                    
                    elapsed = time.time() - run_start_time if run_start_time else 999.0
                    if low_light_active and elapsed <= LOWLIGHT_EVENT_WINDOW_SECONDS:
                        if not shared_data.get('light_event_triggered', False):
                            shared_data['light_event_triggered'] = True
                            print(f"[LIGHT] Light event triggered at {elapsed:.2f}s")
                    
                    if low_light_active and shared_data.get('low_light_recovery_sent', False):
                        time_since_recovery = time.time() - shared_data.get('last_recovery_time', 0)
                        if time_since_recovery > LOWLIGHT_RECOVERY_INTERVAL:
                            shared_data['recovery_phase'] = 1
                
                # Low-brightness fallback rule: if lights are not restored, all tokens behave as yellow.
                if low_light_active:
                    for token in tokens:
                        token['original_color'] = token.get('true_color', token.get('color'))
                        token['true_color'] = 'yellow'
                        token['color'] = 'yellow'
                        token['display_color'] = 'yellow'
                
                tokens = apply_token_visibility_effects(tokens)
                
            except Exception as e:
                print(f"Brightness detection error: {e}")

            with data_lock:
                lane_follow = dict(shared_data.get('lane_follow', {}))
                previous_lane_steering = lane_follow.get('smoothed_steering', 0.0)
                last_lane_follow_time = shared_data.get('last_lane_follow_time', 0.0)
                current_mode = shared_data.get('current_mode', 'LANE FOLLOW')

            now = time.time()
            if now - last_lane_follow_time >= LANE_FOLLOW_INTERVAL_SECONDS:
                lane_follow = detect_lane_following(processed_front_frame, previous_lane_steering)
                with data_lock:
                    shared_data['last_lane_follow_time'] = now

            with data_lock:
                shared_data['detected_tokens'] = tokens
                shared_data['lane_follow'] = lane_follow

            if ENABLE_HEAVY_OVERLAY:
                debug_frame = draw_detected_tokens(processed_front_frame, tokens)
                debug_frame = draw_lane_following_overlay(debug_frame, lane_follow, current_mode)
            else:
                debug_frame = processed_front_frame.copy()
            
            with data_lock:
                if shared_data.get('low_light_active', False):
                    cv2.putText(debug_frame, "⚡ LOW LIGHT - FAST RECOVERY ⚡", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                    cv2.putText(debug_frame, "All tokens YELLOW", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                lights_on = shared_data.get('lights_on', True)
                if not lights_on:
                    cv2.putText(debug_frame, "LIGHTS OFF - tokens become yellow", (10, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                else:
                    cv2.putText(debug_frame, "LIGHTS ON - Normal operation", (10, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                police_active = shared_data.get('police_active', False)
                if police_active:
                    cv2.putText(debug_frame, "🚨 POLICE ACTIVE - SEEK RED TOKEN 🚨", (10, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                trailing_detected = shared_data.get('trailing_detected', False)
                if trailing_detected:
                    cv2.putText(debug_frame, "⚠️ TRAILING CAR - DODGE ⚠️", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            effect = get_active_yellow_effect()
            if effect:
                cv2.putText(debug_frame, f"Yellow Effect: {effect}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if not ENABLE_HEAVY_OVERLAY:
                debug_frame = processed_front_frame.copy()
            debug_frame = draw_light_debug_overlay(debug_frame, tokens, current_mode)
            
            debug_frame = cv2.resize(debug_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            with data_lock:
                shared_data['display_front_frame'] = debug_frame.copy()
        except Exception as e:
            front_status = "Connected - waiting for video stream..." if front_connected else "Not connected yet"
            with data_lock:
                shared_data['display_front_frame'] = make_camera_placeholder("Front Camera", front_status)
    else:
        front_status = "Connected - waiting for video stream..." if front_connected else "Not connected yet"
        with data_lock:
            shared_data['display_front_frame'] = make_camera_placeholder("Front Camera", front_status)

    # Run enhanced trailing and police detection on back camera
    if back_frame is not None:
        try:
            now = time.time()
            with data_lock:
<<<<<<< HEAD
                # Update trailing detection
                if trailing_raw:
                    shared_data['trailing_streak'] = shared_data.get('trailing_streak', 0) + 1
                    shared_data['trailing_lane'] = trailing_lane
                    if bbox is not None:
                        shared_data['trailing_bbox'] = bbox
                        shared_data['trailing_last_seen_time'] = time.time()
                else:
                    shared_data['trailing_streak'] = 0

                # Keep the latest raw police box so the display can draw it after confirmation.
                if police_raw and police_bbox is not None:
                    shared_data['police_bbox'] = police_bbox
                    shared_data['police_last_seen_time'] = time.time()

                # Update police detection
                history = shared_data.get('police_detection_history', [])
                history.append(1 if police_raw else 0)
                if len(history) > POLICE_CONFIRM_FRAMES:
                    history.pop(0)
                shared_data['police_detection_history'] = history
                
                police_confirmed = sum(history) >= (POLICE_CONFIRM_FRAMES * 0.6)
                trailing = shared_data['trailing_streak'] >= TRAILING_CONFIRM_FRAMES
                
                if police_confirmed and police_lane is not None:
                    shared_data['police_lane'] = police_lane
                
                shared_data['trailing_detected'] = trailing
                shared_data['police_detected'] = police_confirmed
                shared_data['danger_detected'] = trailing or police_confirmed
                shared_data['police_confidence'] = confidence
                
                if police_confirmed:
                    shared_data['run_summary']['police_appeared'] = True
                    if not shared_data.get('police_active', False):
                        shared_data['police_active'] = True
                        shared_data['police_start_time'] = time.time()
                        shared_data['red_token_collected_during_police'] = False
                        print(f"[POLICE] Police car confirmed in lane {police_lane}! Confidence: {confidence:.2f}")
                        print(f"[POLICE] Must collect red token within 10 seconds!")
                        if not shared_data.get('police_triggered', False):
                            shared_data['police_triggered'] = True
                
                if trailing:
                    shared_data['run_summary']['trailing_appeared'] = True
                    elapsed = time.time() - run_start_time if run_start_time else 0
                    if not shared_data.get('trailing_first_triggered', False):
                        shared_data['trailing_first_triggered'] = True
                        shared_data['trailing_first_time'] = elapsed
                        shared_data['trailing_count'] = 1
                        print(f"[TRAILING] First trailing car at {elapsed:.2f}s in lane {trailing_lane} (10 sec to avoid)")
                    elif not shared_data.get('trailing_second_triggered', False):
                        shared_data['trailing_second_triggered'] = True
                        shared_data['trailing_second_time'] = elapsed
                        shared_data['trailing_count'] = 2
                        print(f"[TRAILING] Second trailing car at {elapsed:.2f}s in lane {trailing_lane} (only 3 sec to avoid)")
            
            # Draw back camera frame
            dbg = cv2.resize(back_frame.copy(), (DISPLAY_WIDTH, DISPLAY_HEIGHT))

            with data_lock:
                display_trailing_bbox = bbox if bbox is not None else shared_data.get('trailing_bbox')
                display_police_bbox = police_bbox if police_bbox is not None else shared_data.get('police_bbox')
                display_trailing_lane = trailing_lane if trailing_lane is not None else shared_data.get('trailing_lane')
                display_police_lane = police_lane if police_lane is not None else shared_data.get('police_lane')
                trailing_confirmed_display = shared_data.get('trailing_detected', False)
                police_confirmed_display = shared_data.get('police_detected', False)

            if display_trailing_bbox is not None and (trailing_raw or trailing_confirmed_display):
                draw_vehicle_box(
=======
                last_back_detection_time = shared_data.get('last_back_detection_time', 0.0)
                back_debug = dict(shared_data.get('back_detection_debug', {}))

            if now - last_back_detection_time >= BACK_DETECTION_INTERVAL_SECONDS:
                vehicles = []
                trailing_raw, area, bbox, trailing_lane = detect_trailing_car_enhanced(back_frame)
                police_raw, confidence, police_bbox, police_lane = detect_police_car_accurate(back_frame)
                detection_source = "HSV"
                
                back_debug = {
                    'trailing_raw': trailing_raw,
                    'trailing_area': area,
                    'trailing_bbox': bbox,
                    'trailing_lane': trailing_lane,
                    'police_raw': police_raw,
                    'police_confidence': confidence,
                    'police_bbox': police_bbox,
                    'police_lane': police_lane,
                    'vehicles': vehicles,
                    'detection_source': detection_source
                }

                with data_lock:
                    shared_data['last_back_detection_time'] = now
                    shared_data['back_detection_debug'] = back_debug

                    # Update trailing detection only when the heavy detector actually runs.
                    if trailing_raw:
                        shared_data['trailing_streak'] = shared_data.get('trailing_streak', 0) + 1
                        shared_data['trailing_lane'] = trailing_lane
                    else:
                        shared_data['trailing_streak'] = 0
                    
                    # Update police detection
                    history = shared_data.get('police_detection_history', [])
                    history.append(1 if police_raw else 0)
                    if len(history) > POLICE_CONFIRM_FRAMES:
                        history.pop(0)
                    shared_data['police_detection_history'] = history
                    
                    police_confirmed = sum(history) >= (POLICE_CONFIRM_FRAMES * 0.6)
                    trailing = shared_data['trailing_streak'] >= TRAILING_CONFIRM_FRAMES
                    
                    if police_confirmed:
                        shared_data['police_lane'] = police_lane
                    
                    shared_data['trailing_detected'] = trailing
                    shared_data['police_detected'] = police_confirmed
                    shared_data['danger_detected'] = trailing or police_confirmed
                    shared_data['police_confidence'] = confidence
                    
                    if police_confirmed:
                        shared_data['run_summary']['police_appeared'] = True
                        if not shared_data.get('police_active', False):
                            shared_data['police_active'] = True
                            shared_data['police_start_time'] = time.time()
                            shared_data['red_token_collected_during_police'] = False
                            print(f"[POLICE] Police car confirmed in lane {police_lane}! Confidence: {confidence:.2f}")
                            print(f"[POLICE] Must collect red token within 10 seconds!")
                            if not shared_data.get('police_triggered', False):
                                shared_data['police_triggered'] = True
                    
                    if trailing:
                        shared_data['run_summary']['trailing_appeared'] = True
                        elapsed = time.time() - run_start_time if run_start_time else 0
                        if not shared_data.get('trailing_first_triggered', False):
                            shared_data['trailing_first_triggered'] = True
                            shared_data['trailing_first_time'] = elapsed
                            shared_data['trailing_count'] = 1
                            print(f"[TRAILING] First trailing car at {elapsed:.2f}s in lane {trailing_lane} (10 sec to avoid)")
                        elif not shared_data.get('trailing_second_triggered', False):
                            shared_data['trailing_second_triggered'] = True
                            shared_data['trailing_second_time'] = elapsed
                            shared_data['trailing_count'] = 2
                            print(f"[TRAILING] Second trailing car at {elapsed:.2f}s in lane {trailing_lane} (only 3 sec to avoid)")

            trailing_raw = back_debug.get('trailing_raw', False)
            area = back_debug.get('trailing_area', 0.0)
            bbox = back_debug.get('trailing_bbox')
            trailing_lane = back_debug.get('trailing_lane')
            police_raw = back_debug.get('police_raw', False)
            confidence = back_debug.get('police_confidence', 0.0)
            police_bbox = back_debug.get('police_bbox')
            police_lane = back_debug.get('police_lane')
            vehicles = back_debug.get('vehicles', [])
            detection_source = back_debug.get('detection_source', 'HSV')

            if not SHOW_BACK_DEBUG:
                with data_lock:
                    if shared_data.get('police_active', False):
                        police_elapsed = time.time() - shared_data.get('police_start_time', 0)
                        if police_elapsed > 10.0:
                            if not shared_data.get('red_token_collected_during_police', False):
                                print(f"[POLICE] Time expired! No red token collected. Speed reduced by 50%!")
                                if not shared_data.get('police_penalty_applied', False):
                                    shared_data['police_penalty_applied'] = True
                                    shared_data['run_summary']['speed_penalties'] = shared_data['run_summary'].get('speed_penalties', 0) + 1
                            shared_data['police_active'] = False
                dbg = cv2.resize(back_frame.copy(), (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                cv2.putText(dbg, f"Back HSV T:{trailing_raw} P:{police_raw}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                with data_lock:
                    shared_data['display_back_frame'] = dbg.copy()
                return
            
            # Draw back camera frame
            dbg = cv2.resize(back_frame.copy(), (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            scale_x = DISPLAY_WIDTH / back_frame.shape[1]
            scale_y = DISPLAY_HEIGHT / back_frame.shape[0]

            if vehicles:
                scaled_vehicles = []
                for vehicle in vehicles:
                    scaled_vehicle = dict(vehicle)
                    x, y, w, h = vehicle['bbox']
                    scaled_vehicle['bbox'] = (
                        int(x * scale_x),
                        int(y * scale_y),
                        int(w * scale_x),
                        int(h * scale_y)
                    )
                    scaled_vehicles.append(scaled_vehicle)
                dbg = draw_yolo_vehicle_overlay(dbg, scaled_vehicles)
            
            if bbox is not None and trailing_raw:
                x, y, w, h = bbox
                cv2.rectangle(
>>>>>>> 17af060b8a416cdfc3d1217c5a395dad26e52bbf
                    dbg,
                    display_trailing_bbox,
                    back_frame.shape,
                    "TRAILING CAR",
                    lane=display_trailing_lane,
                    color=(0, 255, 255)
                )

            if display_police_bbox is not None and (police_raw or police_confirmed_display):
                draw_vehicle_box(
                    dbg,
                    display_police_bbox,
                    back_frame.shape,
                    "POLICE CAR",
                    lane=display_police_lane,
                    confidence=confidence,
                    color=(0, 0, 255)
                )

            status_color = (0, 255, 0)
            if trailing_raw or trailing_confirmed_display:
                status_color = (0, 255, 255)
            if police_raw or police_confirmed_display:
                status_color = (0, 0, 255)

            lines = [
                "Back Camera",
<<<<<<< HEAD
                f"Trailing: raw={trailing_raw} confirmed={trailing_confirmed_display} Lane: {display_trailing_lane}",
                f"Police: raw={police_raw} confirmed={police_confirmed_display} conf: {confidence:.2f} Lane: {display_police_lane}",
=======
                f"Detector: {detection_source} Vehicles: {len(vehicles)}",
                f"Trailing: {trailing_raw} ({shared_data['trailing_streak']}) Lane: {trailing_lane}",
                f"Police: {police_raw} (conf: {confidence:.2f}) Lane: {police_lane}",
>>>>>>> 17af060b8a416cdfc3d1217c5a395dad26e52bbf
                f"Area: {int(area)}"
            ]

            for idx, line in enumerate(lines):
                cv2.putText(dbg, line, (10, 30 + idx * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Check police time limit
            with data_lock:
                if shared_data.get('police_active', False):
                    police_elapsed = time.time() - shared_data.get('police_start_time', 0)
                    if police_elapsed > 10.0:
                        if not shared_data.get('red_token_collected_during_police', False):
                            print(f"[POLICE] Time expired! No red token collected. Speed reduced by 50%!")
                            if not shared_data.get('police_penalty_applied', False):
                                shared_data['police_penalty_applied'] = True
                                shared_data['run_summary']['speed_penalties'] = shared_data['run_summary'].get('speed_penalties', 0) + 1
                        shared_data['police_active'] = False
            
            with data_lock:
                shared_data['display_back_frame'] = dbg.copy()
        except Exception as e:
            back_status = "Connected - waiting for video stream..." if back_connected else "Not connected yet"
            with data_lock:
                shared_data['display_back_frame'] = make_camera_placeholder("Back Camera", back_status)
    else:
        back_status = "Connected - waiting for video stream..." if back_connected else "Not connected yet"
        with data_lock:
            shared_data['display_back_frame'] = make_camera_placeholder("Back Camera", back_status)

def send_controls_task():
    global control_conn, steering_state, tap_loop_count, current_lane, last_token_lane, last_token_time, locked_green_lane, locked_green_until, committed_target_lane, committed_target_until, committed_target_reason, last_token_debug_time, run_start_time, trailing_dodge_lane, trailing_dodge_until, last_trailing_notice, last_police_notice, locked_green_target, locked_green_target_frames, locked_green_target_missing_frames
    
    if control_conn is None:
        return
    
    with data_lock:
        tokens_snapshot = list(shared_data.get('detected_tokens', []))
        front_frame = shared_data.get('latest_front_frame')
        run_summary = shared_data.get('run_summary', {})
        green_streak = run_summary.get('green_collected', 0)
        red_streak = run_summary.get('red_collected', 0)
        speed_penalties = run_summary.get('speed_penalties', 0)
        trailing_detected = shared_data.get('trailing_detected', False)
        police_detected = shared_data.get('police_detected', False)
        police_active = shared_data.get('police_active', False)
        low_light_active = shared_data.get('low_light_active', False)
        lights_on = shared_data.get('lights_on', True)
        red_token_collected = shared_data.get('red_token_collected_during_police', False)
        police_lane = shared_data.get('police_lane', None)
        trailing_lane = shared_data.get('trailing_lane', None)
        lane_follow = dict(shared_data.get('lane_follow', {}))

    steering_input = 0.0
    acceleration_input = CAR_ACCELERATION 
    desired_lane = current_lane
    decision_reason = "MAINTAIN"
    target_debug = "green:none"
    center_green_steer = None
    green_direct_steer = None
    selected_green_target = None
    predicted_green_x = None
    selected_target_type = "NONE"
    hazard_debug_tokens = []
    speed_modifier = calculate_speed_modifier(green_streak, red_streak, speed_penalties)
    police_elapsed = time.time() - shared_data.get('police_start_time', 0.0) if police_active else 0.0
    police_red_required = police_active and police_elapsed >= POLICE_RED_TARGET_START_SECONDS
    police_debug = {
        'active': police_active,
        'detected': police_detected,
        'elapsed': police_elapsed,
        'remaining': max(0.0, POLICE_EVENT_TIMEOUT_SECONDS - police_elapsed) if police_active else 0.0,
        'red_required': police_red_required,
        'reason': 'inactive'
    }

    def apply_speed_rules(raw_acceleration):
        if raw_acceleration <= 0.0:
            return raw_acceleration
        return max(0.0, min(MAX_ACCELERATION, raw_acceleration * speed_modifier))

    if run_start_time is not None and time.time() - run_start_time < START_CENTER_HOLD_SECONDS:
        try: 
            control_conn.sendall(struct.pack('ff', 0.0, 0.0))
        except Exception: 
            pass
        current_lane = START_LANE
        return

    img_w, img_h = (front_frame.shape[1], front_frame.shape[0]) if front_frame is not None else (640, 480)
    
<<<<<<< HEAD
    # Detect hazard tokens to avoid.
    # Red is always a hazard except during police-active red-token seeking.
    # Yellow is always a hazard because it triggers negative/random effects.
    red_tokens = [
        t for t in tokens_snapshot
        if t.get('color') == 'red' and t['y'] >= TOKEN_DECISION_Y_MIN
    ]
    yellow_tokens = [
        t for t in tokens_snapshot
        if t.get('color') == 'yellow' and t['y'] >= img_h * YELLOW_HAZARD_Y_MIN_RATIO
    ]

    hazard_lanes = set()
    yellow_hazard_lanes = set()

    for t in red_tokens:
        t_lane = lane_from_x(t['x'], t['y'], frame_width=img_w, frame_height=img_h)
        # During police-active mode, red can be a target. Otherwise it is a blocked lane.
        if not police_active:
            hazard_lanes.add(t_lane)

    for t in yellow_tokens:
        t_lane = lane_from_x(t['x'], t['y'], frame_width=img_w, frame_height=img_h)
        hazard_lanes.add(t_lane)
        yellow_hazard_lanes.add(t_lane)

        # If yellow is very close, also avoid the neighbouring lane to reduce side-swipe collection.
        if t['y'] >= img_h * YELLOW_ADJACENT_BUFFER_Y_RATIO:
            if t_lane - 1 >= -2:
                hazard_lanes.add(t_lane - 1)
                yellow_hazard_lanes.add(t_lane - 1)
            if t_lane + 1 <= 2:
                hazard_lanes.add(t_lane + 1)
                yellow_hazard_lanes.add(t_lane + 1)

    # Bonus lanes must be green only.
    # Do not use yellow as a bonus lane during evasion, because that makes the car drift into yellow tokens.
    green_tokens_for_bonus = [
        t for t in tokens_snapshot
        if t.get('color') == 'green' and t['y'] >= TOKEN_DECISION_Y_MIN
    ]
    bonus_lanes = set()
    for t in green_tokens_for_bonus:
        t_lane = lane_from_x(t['x'], t['y'], frame_width=img_w, frame_height=img_h)
        if t_lane not in hazard_lanes:
            bonus_lanes.add(t_lane)
=======
    # Avoid red/yellow during normal driving. Red is only a target while Police is active.
    hazard_colors = {'yellow'} if police_red_required else {'red', 'yellow'}
    hazard_tokens = [
        t for t in tokens_snapshot
        if t.get('color') in hazard_colors and t['y'] >= HAZARD_AVOID_Y_MIN
    ]
    red_hazard_tokens = [
        t for t in tokens_snapshot
        if not police_red_required and t.get('color') == 'red' and t['y'] >= HAZARD_AVOID_Y_MIN
    ]
    yellow_hazard_tokens = [
        t for t in tokens_snapshot
        if t.get('color') == 'yellow' and t['y'] >= HAZARD_AVOID_Y_MIN
    ]
    hazard_lanes = set()
    close_hazard_lanes = set()
    for t in hazard_tokens:
        t_lane = lane_from_x(t['x'], t['y'], frame_width=img_w, frame_height=img_h)
        hazard_lanes.add(t_lane)
        if t['y'] >= GREEN_BLOCKED_HAZARD_Y_MIN:
            close_hazard_lanes.add(t_lane)
    red_hazard_lanes = set()
    for t in red_hazard_tokens:
        red_hazard_lanes.add(lane_from_x(t['x'], t['y'], frame_width=img_w, frame_height=img_h))
    yellow_hazard_lanes = set()
    for t in yellow_hazard_tokens:
        yellow_hazard_lanes.add(lane_from_x(t['x'], t['y'], frame_width=img_w, frame_height=img_h))

    # Green lanes are useful escape lanes; yellow is not a bonus under the distance/survival objective.
    green_tokens_for_bonus = [t for t in tokens_snapshot if t.get('color') == 'green' and t['y'] >= TOKEN_DECISION_Y_MIN]
    bonus_tokens = green_tokens_for_bonus
    bonus_lanes = set()
    for t in bonus_tokens:
        t_lane = lane_from_x(t['x'], t['y'], frame_width=img_w, frame_height=img_h)
        bonus_lanes.add(t_lane)
    green_tokens_visible = [
        t for t in tokens_snapshot
        if t.get('color') == 'green'
    ]
    green_reject_counts = {
        'candidate': len(green_tokens_visible),
        'too_far': 0,
        'outside_roi': 0,
        'blocked_by_hazard': 0,
        'event_override': 0,
        'no_lock': 0,
        'unreachable_lane': 0
    }
    for token in green_tokens_visible:
        reason = green_rejection_reason(token, img_w, img_h, current_lane, hazard_lanes)
        if reason:
            green_reject_counts[reason] = green_reject_counts.get(reason, 0) + 1
>>>>>>> 17af060b8a416cdfc3d1217c5a395dad26e52bbf

    # =========================================================
    # EMERGENCY EVASION - Police car in same lane (front)
    # =========================================================
    if police_red_required and (not trailing_detected or trailing_lane != current_lane) and police_detected and police_lane is not None and police_lane == current_lane:
        print(f"[POLICE] ⚠️ Police car in lane {police_lane}! EVADING!")
        # Find safe lane to move to
        safe_lane = get_safe_lane(current_lane, hazard_lanes, img_w, bonus_lanes)
        if safe_lane != current_lane:
            desired_lane = safe_lane
            decision_reason = "POLICE_EVADE"
            selected_target_type = "POLICE"
            acceleration_input = min(acceleration_input, 0.78)
            with data_lock:
                shared_data['target_lane'] = safe_lane

    # =========================================================
    # EMERGENCY EVASION - Trailing car in same lane (back)
    # =========================================================
    if trailing_detected and trailing_lane is not None and trailing_lane == current_lane:
        print(f"[TRAILING] ⚠️ Trailing car in lane {trailing_lane}! EVADING!")
        # Track trailing event timing
        with data_lock:
            elapsed = time.time() - run_start_time if run_start_time else 0
            if shared_data.get('trailing_first_triggered', False) and not shared_data.get('trailing_second_triggered', False):
                # First trailing car - 10 seconds to avoid
                time_since_first = elapsed - shared_data.get('trailing_first_time', 0)
                if time_since_first > 10.0:
                    print(f"[TRAILING] ⚠️ Time expired! Speed reduced by 50%!")
                    if not shared_data.get('trailing_first_penalty_applied', False):
                        shared_data['trailing_first_penalty_applied'] = True
                        shared_data['run_summary']['speed_penalties'] = shared_data['run_summary'].get('speed_penalties', 0) + 1
                        speed_penalties += 1
                        speed_modifier = calculate_speed_modifier(green_streak, red_streak, speed_penalties)
            elif shared_data.get('trailing_second_triggered', False):
                # Second trailing car - only 3 seconds to avoid
                time_since_second = elapsed - shared_data.get('trailing_second_time', 0)
                if time_since_second > 3.0:
                    print(f"[TRAILING] ⚠️ Time expired! Speed reduced by 50%!")
                    if not shared_data.get('trailing_second_penalty_applied', False):
                        shared_data['trailing_second_penalty_applied'] = True
                        shared_data['run_summary']['speed_penalties'] = shared_data['run_summary'].get('speed_penalties', 0) + 1
                        speed_penalties += 1
                        speed_modifier = calculate_speed_modifier(green_streak, red_streak, speed_penalties)
        
        # Find safe lane to move to
        safe_lane = get_safe_lane(current_lane, hazard_lanes, img_w, bonus_lanes)
        if safe_lane != current_lane:
            desired_lane = safe_lane
            decision_reason = "TRAILING_EVADE"
            selected_target_type = "TRAILING"
            acceleration_input = min(acceleration_input, 0.78)
            with data_lock:
                shared_data['target_lane'] = safe_lane

    # =========================================================
    # POLICE CAR HANDLING - Seek red token before low-light/yellow hazards.
    # =========================================================
    if police_active and decision_reason == "MAINTAIN":
        red_tokens_police = [t for t in tokens_snapshot if t.get('color') == 'red' and t['y'] >= TOKEN_DECISION_Y_MIN]
        police_debug['reason'] = 'waiting_until_timeout'
        if police_red_required and red_tokens_police:
            target_red = max(red_tokens_police, key=lambda t: (t['y'], -abs(t['x'] - (img_w / 2.0))))
            red_lane = lane_from_x(target_red['x'], target_red['y'], frame_width=img_w, frame_height=img_h)
            desired_lane = max(-2, min(2, red_lane))
            decision_reason = "POLICE_RED_TOKEN"
            selected_target_type = "RED"
            acceleration_input = min(acceleration_input, POLICE_SEEK_ACCELERATION)
            police_debug['reason'] = 'near_timeout_target_red'
            target_debug = f"police_red:{target_red['x']},{target_red['y']} left:{max(0.0, POLICE_EVENT_TIMEOUT_SECONDS - police_elapsed):.1f}s"
        elif police_elapsed > POLICE_EVENT_TIMEOUT_SECONDS:
            if not red_token_collected:
                print(f"[POLICE] Time expired! No red token collected. Speed reduced by 50%!")
                with data_lock:
                    if not shared_data.get('police_penalty_applied', False):
                        shared_data['police_penalty_applied'] = True
                        shared_data['run_summary']['speed_penalties'] = shared_data['run_summary'].get('speed_penalties', 0) + 1
                        speed_penalties += 1
                        speed_modifier = calculate_speed_modifier(green_streak, red_streak, speed_penalties)
            shared_data['police_active'] = False
            police_debug['active'] = False
            police_debug['reason'] = 'expired'
        else:
            if police_red_required:
                police_debug['reason'] = 'near_timeout_no_red_visible'
            else:
                police_debug['reason'] = 'active_not_urgent'

    # =========================================================
    # SUPER FAST LIGHT RECOVERY
    # =========================================================
    if low_light_active:
        with data_lock:
            recovery_sent = shared_data.get('low_light_recovery_sent', False)
            last_recovery_time = shared_data.get('last_recovery_time', 0.0)
            recovery_attempts = shared_data.get('recovery_attempts', 0)
            current_brightness = shared_data.get('brightness_last', 0.0)
            baseline = shared_data.get('brightness_baseline', current_brightness)
        
        # Check if light is restored
        if current_brightness > baseline * LOWLIGHT_RECOVERY_THRESHOLD:
            print(f"[LIGHT] ✅ Light restored! Brightness: {current_brightness:.1f}")
            with data_lock:
                shared_data['low_light_active'] = False
                shared_data['lights_on'] = True
                shared_data['low_light_recovery_sent'] = False
                shared_data['recovery_completed'] = True
                shared_data['recovery_phase'] = 0
                shared_data['run_summary']['light_recovered'] = True
            return
        
        # Send recovery signal quickly
        time_since_recovery = time.time() - last_recovery_time
        
        if not recovery_sent or time_since_recovery > LOWLIGHT_RECOVERY_INTERVAL:
            steering_input = 0.0
            acceleration_input = -1.0
            
            with data_lock:
                shared_data['low_light_recovery_sent'] = True
                shared_data['last_recovery_time'] = time.time()
                shared_data['recovery_attempts'] = recovery_attempts + 1
                shared_data['run_summary']['light_recovery_attempts'] = shared_data['run_summary'].get('light_recovery_attempts', 0) + 1
                shared_data['steering_input'] = steering_input
                shared_data['acceleration_input'] = acceleration_input
                shared_data['speed_modifier'] = speed_modifier
                shared_data['selected_target_type'] = 'LOW_LIGHT'
                shared_data['current_mode'] = 'LOW_LIGHT'
            
            print(f"[LIGHT] ⚡ FAST RECOVERY (attempt {recovery_attempts + 1})")
            
            # Send fast recovery
            for i in range(LOWLIGHT_FAST_RECOVERY_COUNT):
                try:
                    send_control_packet(0.0, -1.0)
                    time.sleep(0.03)
                except Exception:
                    pass
            return
        
        # While waiting, maintain minimal control
        steering_input = 0.0
        acceleration_input = apply_speed_rules(MIN_ACCELERATION_WHEN_SLOWED)
        with data_lock:
            shared_data['steering_input'] = steering_input
            shared_data['acceleration_input'] = acceleration_input
            shared_data['speed_modifier'] = speed_modifier
            shared_data['selected_target_type'] = 'LOW_LIGHT'
            shared_data['current_mode'] = 'LOW_LIGHT'
        
        try:
            send_control_packet(steering_input, acceleration_input)
        except Exception as e:
            print(f"Network error: {e}")
            control_conn = None
        return

    if decision_reason == "MAINTAIN" and current_lane in red_hazard_lanes:
        safe_lane = get_safe_lane(current_lane, hazard_lanes, img_w, bonus_lanes)
        if safe_lane != current_lane:
            desired_lane = safe_lane
            decision_reason = "AVOID_RED"
            selected_target_type = "AVOID_RED"
            hazard_debug_tokens = red_hazard_tokens
            acceleration_input = min(acceleration_input, 0.80)

    if decision_reason == "MAINTAIN" and current_lane in yellow_hazard_lanes:
        safe_lane = get_safe_lane(current_lane, hazard_lanes, img_w, bonus_lanes)
        if safe_lane != current_lane:
            desired_lane = safe_lane
            decision_reason = "AVOID_YELLOW"
            selected_target_type = "AVOID_YELLOW"
            hazard_debug_tokens = yellow_hazard_tokens
            acceleration_input = min(acceleration_input, 0.80)

    # =========================================================
    # TOKEN DECISION - Seek reachable GREEN tokens after hazards
    # =========================================================
<<<<<<< HEAD
    if not police_active and not police_detected and not trailing_detected:
        danger_tokens_front = [
            t for t in tokens_snapshot
            if t.get('color') in ['red', 'yellow']
            and token_in_front_danger_zone(t, img_w, img_h)
        ]

        if danger_tokens_front:
            closest_danger = max(danger_tokens_front, key=lambda t: t['y'])

            steering_input = -1.0 if closest_danger['x'] >= img_w / 2.0 else 1.0
            acceleration_input = HAZARD_AVOID_MAX_ACCELERATION

            with data_lock:
                shared_data['decision_debug'] = f"FRONT DANGER OVERRIDE: {closest_danger['color']}"

            send_control_packet(steering_input, acceleration_input)
            return
    # Only seek green tokens if no police or trailing car detected
    # =========================================================
# TOKEN DECISION - GREEN PRIORITY WITH HAZARD AWARENESS
# =========================================================

# Only seek green tokens if no police or trailing car detected
    if not police_detected and not trailing_detected:

        # -----------------------------
        # STEP 1: COLLECT ONLY GREEN TARGETS
        # -----------------------------
        green_tokens = [
            t for t in tokens_snapshot
            if t.get('color') == 'green' and t['y'] >= TOKEN_DECISION_Y_MIN
        ]

        hazard_tokens = [
            t for t in tokens_snapshot
            if t.get('color') in ['red', 'yellow'] and t['y'] >= TOKEN_DECISION_Y_MIN
        ]

        # -----------------------------
        # STEP 2: LANE SCORING SYSTEM
        # -----------------------------
        lane_density = {lane: 0.0 for lane in range(-2, 3)}
        lane_closest_token = {}

        for t in green_tokens:
            t_lane = lane_from_x(t['x'], t['y'], frame_width=img_w, frame_height=img_h)

            # Never chase a green token if its lane is blocked by red/yellow.
            if t_lane in hazard_lanes or t_lane in yellow_hazard_lanes:
                continue

            proximity = t['y'] / img_h

            # Reward green tokens, especially closer ones.
            lane_density[t_lane] += 1.0 + (proximity * 2.0)

            if t_lane not in lane_closest_token or t['y'] > lane_closest_token[t_lane]['y']:
                lane_closest_token[t_lane] = t

        # -----------------------------
        # STEP 3: APPLY STRONG HAZARD PENALTY
        # -----------------------------
        for h in hazard_tokens:
            h_lane = lane_from_x(h['x'], h['y'], frame_width=img_w, frame_height=img_h)
            proximity = h['y'] / img_h

            if h_lane in lane_density:
                if h.get('color') == 'red':
                    lane_density[h_lane] -= 8.0 + (proximity * 6.0)
                elif h.get('color') == 'yellow':
                    # Yellow must be heavily penalised to reduce yellow collection.
                    lane_density[h_lane] -= YELLOW_LANE_BLOCK_SCORE + (proximity * 20.0)

                    # If yellow is close, treat neighbouring lanes as risky too.
                    if proximity >= YELLOW_CLOSE_Y_RATIO:
                        if h_lane - 1 in lane_density:
                            lane_density[h_lane - 1] -= 10.0
                        if h_lane + 1 in lane_density:
                            lane_density[h_lane + 1] -= 10.0

        # -----------------------------
        # STEP 4: CHOOSE BEST SAFE LANE
        # -----------------------------
        valid_lanes = [
            l for l in lane_density.keys()
            if lane_density[l] > -5 and l not in yellow_hazard_lanes
        ]

        if valid_lanes:
            best_lane = max(lane_density, key=lane_density.get)

            if lane_density[best_lane] == 0:
                best_target_lane = current_lane   
            else:
                best_target_lane = best_lane
        else:
            best_target_lane = get_safe_lane(current_lane, hazard_lanes, img_w, set())

        target = lane_closest_token.get(best_target_lane, None)

        # -----------------------------
        # STEP 5: STEERING DECISION
        # -----------------------------
        if target is not None and best_target_lane not in hazard_lanes:
            token_error = (target['x'] - (img_w / 2.0)) / (img_w / 2.0)

            if abs(token_error) > TOKEN_CENTER_X_DEADZONE:
                steering_input = max(-1.0, min(1.0, token_error * GREEN_STEER_GAIN))

                if abs(steering_input) < GREEN_APPROACH_MIN_STEER:
                    steering_input = GREEN_APPROACH_MIN_STEER if steering_input > 0 else -GREEN_APPROACH_MIN_STEER
            else:
                if best_target_lane > current_lane:
                    steering_input = 1.0
                elif best_target_lane < current_lane:
                    steering_input = -1.0
                else:
                    steering_input = 0.0

            acceleration_input = min(acceleration_input, GREEN_CHASE_ACCELERATION)

        else:
            # No safe green target. Move away from yellow/red instead of chasing tokens.
            safe_lane = get_safe_lane(current_lane, hazard_lanes, img_w, set())

            if safe_lane != current_lane:
                steering_input = 1.0 if safe_lane > current_lane else -1.0
                acceleration_input = min(acceleration_input, 0.58)
            else:
                steering_input = 0.0
                acceleration_input = min(CAR_ACCELERATION, 0.66)
=======
    if decision_reason != "MAINTAIN" or trailing_detected:
        green_reject_counts['event_override'] = len(green_tokens_visible)
    if decision_reason == "MAINTAIN" and not trailing_detected:
        previous_target = locked_green_target if locked_green_target_frames > 0 else None
        selected_green_target, reachable_green_tokens = select_green_target(
            green_tokens_visible,
            previous_target,
            img_w,
            img_h,
            current_lane,
            hazard_lanes
        )

        if selected_green_target is not None:
            target_is_locked = False
            if previous_target:
                dx = selected_green_target['x'] - previous_target.get('x', selected_green_target['x'])
                dy = selected_green_target['y'] - previous_target.get('y', selected_green_target['y'])
                lock_radius = max(45.0, img_w * 0.11)
                target_is_locked = (dx * dx + dy * dy) <= lock_radius * lock_radius

            locked_green_target = dict(selected_green_target)
            if target_is_locked:
                locked_green_target_frames = max(1, locked_green_target_frames - 1)
            else:
                locked_green_target_frames = GREEN_TARGET_LOCK_FRAMES
            locked_green_target_missing_frames = 0
            target_lane = lane_from_x(
                selected_green_target['x'],
                selected_green_target['y'],
                frame_width=img_w,
                frame_height=img_h
            )
            desired_lane = max(-2, min(2, target_lane))
            target_y_ratio = selected_green_target['y'] / float(img_h)
            decision_reason = "GREEN_TARGET" if target_y_ratio >= GREEN_COLLECT_Y_RATIO else "PRE_TARGET_GREEN"
            selected_target_type = "GREEN"
            acceleration_input = max(acceleration_input, GREEN_CHASE_ACCELERATION)

            image_center_x = img_w / 2.0
            prediction_scale = 1.0 + max(0.0, 1.0 - target_y_ratio) * GREEN_PREDICTION_GAIN
            predicted_green_x = clamp(
                image_center_x + ((selected_green_target['x'] - image_center_x) * prediction_scale),
                0.0,
                float(img_w - 1)
            )
            green_error = predicted_green_x - image_center_x
            normalized_error = green_error / image_center_x
            if abs(normalized_error) <= GREEN_CENTER_DEADZONE_RATIO:
                green_direct_steer = 0.0
            else:
                steer_gain = GREEN_DIRECT_STEER_GAIN if decision_reason == "GREEN_TARGET" else GREEN_PRE_TARGET_STEER_GAIN
                green_direct_steer = clamp(normalized_error * steer_gain, -1.0, 1.0)
            target_debug = (
                f"green_target:{selected_green_target['x']},{selected_green_target['y']} "
                f"pred:{int(predicted_green_x)} reach:{len(reachable_green_tokens)} lock:{locked_green_target_frames}"
            )
        else:
            if previous_target and green_rejection_reason(previous_target, img_w, img_h, current_lane, hazard_lanes) is None and locked_green_target_missing_frames < GREEN_TARGET_MAX_MISSING_FRAMES:
                locked_green_target_missing_frames += 1
                locked_green_target_frames = max(1, locked_green_target_frames - 1)
                selected_green_target = dict(previous_target)
                target_lane = lane_from_x(selected_green_target['x'], selected_green_target['y'], frame_width=img_w, frame_height=img_h)
                desired_lane = max(-2, min(2, target_lane))
                target_y_ratio = selected_green_target['y'] / float(img_h)
                decision_reason = "GREEN_TARGET" if target_y_ratio >= GREEN_COLLECT_Y_RATIO else "PRE_TARGET_GREEN"
                selected_target_type = "GREEN"
                acceleration_input = max(acceleration_input, GREEN_CHASE_ACCELERATION)
                image_center_x = img_w / 2.0
                prediction_scale = 1.0 + max(0.0, 1.0 - target_y_ratio) * GREEN_PREDICTION_GAIN
                predicted_green_x = clamp(image_center_x + ((selected_green_target['x'] - image_center_x) * prediction_scale), 0.0, float(img_w - 1))
                normalized_error = (predicted_green_x - image_center_x) / image_center_x
                steer_gain = GREEN_DIRECT_STEER_GAIN if decision_reason == "GREEN_TARGET" else GREEN_PRE_TARGET_STEER_GAIN
                green_direct_steer = 0.0 if abs(normalized_error) <= GREEN_CENTER_DEADZONE_RATIO else clamp(normalized_error * steer_gain, -1.0, 1.0)
                target_debug = f"green_lock_missing:{locked_green_target_missing_frames}/{GREEN_TARGET_MAX_MISSING_FRAMES} x:{selected_green_target['x']} y:{selected_green_target['y']}"
            else:
                locked_green_target = None
                locked_green_target_frames = 0
                locked_green_target_missing_frames = 0
                green_reject_counts['no_lock'] = len(green_tokens_visible)
            if green_tokens_visible:
                safe_lane = get_safe_lane(current_lane, hazard_lanes, img_w, set())
                if decision_reason == "MAINTAIN" and safe_lane != current_lane:
                    desired_lane = safe_lane
                    decision_reason = "GREEN_BLOCKED_AVOID"
                    selected_target_type = "HAZARD"
                    acceleration_input = min(acceleration_input, 0.78)
>>>>>>> 17af060b8a416cdfc3d1217c5a395dad26e52bbf

    if decision_reason not in ("GREEN_TARGET", "PRE_TARGET_GREEN") and locked_green_target_frames > 0:
        locked_green_target_frames = 0
        locked_green_target = None
        locked_green_target_missing_frames = 0

    if green_direct_steer is not None:
        steering_input = green_direct_steer
    else:
        steering_input = apply_tap_steering(desired_lane)

    if decision_reason == "MAINTAIN" and steering_state == 0:
        lane_center_x = lane_follow.get('lane_center_x')
        if lane_follow.get('valid') and lane_center_x is not None:
            steering_input = float(lane_follow.get('smoothed_steering', 0.0))
            decision_reason = "LANE_FOLLOW"
            target_debug = f"lane_center:{lane_center_x:.1f}"

    acceleration_input = apply_speed_rules(acceleration_input)

    mode_map = {
        "POLICE_EVADE": "POLICE MODE",
        "POLICE_RED_TOKEN": "POLICE MODE",
        "TRAILING_EVADE": "TRAILING CAR AVOID",
        "PRE_TARGET_GREEN": "PRE_TARGET_GREEN",
        "GREEN_TARGET": "COLLECT_GREEN",
        "GREEN_BLOCKED_AVOID": "AVOID RED",
        "AVOID_RED": "AVOID RED",
        "AVOID_YELLOW": "AVOID YELLOW",
        "LANE_FOLLOW": "LANE FOLLOW"
    }
    if trailing_detected and decision_reason == "MAINTAIN":
        current_mode_text = "TRAILING CAR AVOID"
    else:
        current_mode_text = mode_map.get(decision_reason, "LANE FOLLOW")

    with data_lock:
<<<<<<< HEAD
        shared_data['decision_debug'] = f"Police: {police_detected} Lane:{police_lane} | Trailing: {trailing_detected} Lane:{trailing_lane} | YellowHaz:{sorted(list(yellow_hazard_lanes))} | Steer: {steering_input:.2f} | Accel: {acceleration_input:.2f}"
=======
        shared_data['target_lane'] = desired_lane
        shared_data['current_mode'] = current_mode_text
        shared_data['decision_debug'] = f"{decision_reason} {target_debug} L{current_lane}->{desired_lane} T{steering_state} S{steering_input:.2f} A{acceleration_input:.2f}"
        shared_data['steering_input'] = steering_input
        shared_data['acceleration_input'] = acceleration_input
        shared_data['speed_modifier'] = speed_modifier
        shared_data['selected_target_type'] = selected_target_type
        shared_data['hazard_debug'] = {'tokens': hazard_debug_tokens} if hazard_debug_tokens else None
        shared_data['police_debug'] = dict(police_debug)
        shared_data['green_decision_debug'] = {
            'candidate_count': green_reject_counts.get('candidate', 0),
            'lock_frames': locked_green_target_frames,
            'target_age': max(0, GREEN_TARGET_LOCK_FRAMES - locked_green_target_frames),
            'missing_frames': locked_green_target_missing_frames,
            'reject': dict(green_reject_counts)
        }
        if selected_green_target is not None and decision_reason in ("GREEN_TARGET", "PRE_TARGET_GREEN"):
            shared_data['green_target_debug'] = {
                'x': int(selected_green_target['x']),
                'y': int(selected_green_target['y']),
                'predicted_x': int(predicted_green_x) if predicted_green_x is not None else int(selected_green_target['x']),
                'lock_frames': locked_green_target_frames,
                'target_age': max(0, GREEN_TARGET_LOCK_FRAMES - locked_green_target_frames)
            }
        else:
            shared_data['green_target_debug'] = None
>>>>>>> 17af060b8a416cdfc3d1217c5a395dad26e52bbf

    try:
        send_control_packet(steering_input, acceleration_input)
    except Exception as e:
        print(f"Network error: {e}")
        control_conn = None

# ---------------------------------------------------------
# Main (Scheduler Initialization)
# ---------------------------------------------------------
if __name__ == '__main__':
    locked_green_target = None
    locked_green_target_frames = 0
    locked_green_target_missing_frames = 0
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
    run_start_time = time.time()
    
    with data_lock:
        shared_data['target_lane'] = START_LANE
        shared_data['brightness_baseline'] = None
        shared_data['brightness_last'] = 0.0
        shared_data['low_light_active'] = False
        shared_data['low_light_recovered'] = False
        shared_data['low_light_recovery_sent'] = False
        shared_data['trailing_streak'] = 0
        shared_data['police_streak'] = 0
        shared_data['lights_on'] = True
        shared_data['light_event_triggered'] = False
        shared_data['trailing_first_triggered'] = False
        shared_data['trailing_second_triggered'] = False
        shared_data['police_triggered'] = False
        shared_data['recovery_attempts'] = 0
        shared_data['last_recovery_time'] = 0.0
        shared_data['recovery_completed'] = False
        shared_data['recovery_phase'] = 0
        shared_data['trailing_count'] = 0
        shared_data['trailing_first_time'] = 0.0
        shared_data['trailing_second_time'] = 0.0
        shared_data['police_active'] = False
        shared_data['police_start_time'] = 0.0
        shared_data['red_token_collected_during_police'] = False
        shared_data['police_penalty_applied'] = False
        shared_data['trailing_first_penalty_applied'] = False
        shared_data['trailing_second_penalty_applied'] = False
        shared_data['police_confidence'] = 0.0
        shared_data['police_detection_history'] = []
        shared_data['police_lane'] = None
        shared_data['trailing_lane'] = None
<<<<<<< HEAD
        shared_data['police_bbox'] = None
        shared_data['trailing_bbox'] = None
        shared_data['police_last_seen_time'] = 0.0
        shared_data['trailing_last_seen_time'] = 0.0
=======
        shared_data['lane_follow'] = {
            'lane_center_x': None,
            'left_line': None,
            'right_line': None,
            'raw_steering': 0.0,
            'smoothed_steering': 0.0,
            'valid': False
        }
        shared_data['last_lane_follow_time'] = 0.0
        shared_data['last_back_detection_time'] = 0.0
        shared_data['back_detection_debug'] = {
            'trailing_raw': False,
            'trailing_area': 0.0,
            'trailing_bbox': None,
            'trailing_lane': None,
            'police_raw': False,
            'police_confidence': 0.0,
            'police_bbox': None,
            'police_lane': None
        }
        shared_data['green_target_debug'] = None
        shared_data['green_decision_debug'] = None
        shared_data['police_debug'] = None
        shared_data['hazard_debug'] = None
        shared_data['speed_modifier'] = 1.0
        shared_data['selected_target_type'] = 'NONE'
        shared_data['current_mode'] = 'LANE FOLLOW'
>>>>>>> 17af060b8a416cdfc3d1217c5a395dad26e52bbf
        shared_data['run_summary']['light_recovery_attempts'] = 0
        shared_data['run_summary']['light_recovered'] = False
        shared_data['run_summary']['speed_penalties'] = 0

    cv2.namedWindow("Front Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Back Camera", cv2.WINDOW_NORMAL)
    with data_lock:
        shared_data['display_front_frame'] = make_camera_placeholder("Front Camera", "Starting controller...")
        shared_data['display_back_frame'] = make_camera_placeholder("Back Camera", "Starting controller...")
    cv2.moveWindow("Front Camera", 40, 40)
    cv2.moveWindow("Back Camera", 700, 40)
    update_camera_windows()

    print("\n" + "="*60)
    print(" 🏁 SPEED TRIALS 2D - FULLY AUTONOMOUS DRIVER 🏁")
    print("="*60)
    print("\nFeatures Implemented:")
    print("  - Super Fast Light Detection & Recovery (0.2s interval)")
    print("  - Police Car Detection (Split Red/Blue) with Lane Tracking")
    print("  - Trailing Car Detection (Cyan/Teal) with Lane Tracking")
    print("  - Emergency Lane Evasion for Same-Lane Threats")
    print("  - Green Token Seeking (Avoid Red Tokens)")
    print("  - Split Screen Detection (Not Low Light)")
    print("\nInitializing...")
    
    threading.Thread(target=setup_control_server, daemon=True).start()
    threading.Thread(target=setup_cameras, daemon=True).start()
    
    print("\n--- Starting Real-Time Tasks ---\n")
    
    t_front_camera = RTTask("ReadFrontCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_front_camera_task)
    t_back_camera = RTTask("ReadBackCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_back_camera_task)
    t_processing = RTTask("Processing", period=PROCESSING_PERIOD_SECONDS, priority=TaskPriority.MEDIUM, execute_func=processing_task)
    t_controls = RTTask("SendControls", period=0.005, priority=TaskPriority.HIGH, execute_func=send_controls_task)
    
    t_front_camera.start()
    t_back_camera.start()
    t_processing.start()
    t_controls.start()

    try:
        while is_running:
            update_camera_windows()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Stopping system...")
        is_running = False

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
    
    print("\nSystem terminated cleanly.")

    print("\n" + "="*60)
    print(" 🚗  END OF RUN SUMMARY  🚗")
    print("="*60)
    with data_lock:
        summary = shared_data.get('run_summary', {})
        events = {
            'light_event': shared_data.get('light_event_triggered', False),
            'trailing_first': shared_data.get('trailing_first_triggered', False),
            'trailing_second': shared_data.get('trailing_second_triggered', False),
            'police_event': shared_data.get('police_triggered', False)
        }
        light_attempts = shared_data.get('recovery_attempts', 0)
        light_recovered = summary.get('light_recovered', False)
        trailing_count = shared_data.get('trailing_count', 0)
        police_confidence = shared_data.get('police_confidence', 0.0)
    
    if summary:
        print("\nTokens Collected:")
        print(f"  🟢 Green:  {summary.get('green_collected', 0)}")
        print(f"  🟡 Yellow: {summary.get('yellow_collected', 0)}")
        print(f"  🔴 Red:    {summary.get('red_collected', 0)}")
        
        yellow_effects = summary.get('yellow_effects', {})
        if yellow_effects:
            print("\nYellow Effects Triggered:")
            for effect_name, effect_count in yellow_effects.items():
                print(f"  - {effect_name}: {effect_count}")
        print(f"\nSpeed Penalties Applied: {summary.get('speed_penalties', 0)}")
        
        print("\n" + "-"*60)
        print("EVENT STATUS:")
        print("-"*60)
        print(f"  Light Event:              {'✓ DETECTED & RECOVERED' if light_recovered else ('✓ DETECTED' if events['light_event'] else '✗ NOT TRIGGERED')}")
        if light_attempts > 0:
            print(f"    - Recovery attempts: {light_attempts}")
        print(f"  Trailing Cars:            {trailing_count} detected")
        if events['trailing_first']:
            print(f"    - First: ✓ (10s to avoid)")
        if events['trailing_second']:
            print(f"    - Second: ✓ (3s to avoid)")
        print(f"  Police Car:               {'✓ DETECTED' if events['police_event'] else '✗ NOT DETECTED'}")
        if events['police_event']:
            print(f"    - Confidence: {police_confidence:.2f}")
            print(f"    - Red token collected: {'✓ YES' if shared_data.get('red_token_collected_during_police', False) else '✗ NO (speed penalty applied)'}")
        
        print("\n" + "-"*60)
        print("DETECTED EVENTS:")
        print("-"*60)
        print(f"  🚓 Police Car Appeared:   {'YES' if summary.get('police_appeared') else 'NO'}")
        print(f"  🚘 Trailing Car Appeared: {'YES' if summary.get('trailing_appeared') else 'NO'}")
    
    print("\n" + "="*60 + "\n")
