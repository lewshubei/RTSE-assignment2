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
GREEN_CHASE_ACCELERATION = 0.98
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
LANE_CHANGE_TAP_LOOPS = 8
LANE_CHANGE_RESET_LOOPS = 2
TOKEN_TARGET_MEMORY_SECONDS = 2.4
GREEN_TARGET_LOCK_SECONDS = 2.8
GREEN_TARGET_LOCK_FRAMES = 6
GREEN_TARGET_MAX_MISSING_FRAMES = 5
GREEN_DECISION_MIN_Y_RATIO = 0.10
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
HAZARD_AVOID_Y_MIN = 35
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

# Light detection tunables - SUPER FAST RECOVERY
LOWLIGHT_DARK_THRESHOLD = 0.45  # Lowered for faster detection
LOWLIGHT_RECOVERY_THRESHOLD = 0.60  # Lowered for faster recovery
LOWLIGHT_MIN_BASELINE = 25.0
LOWLIGHT_EVENT_WINDOW_SECONDS = 10.0
LOWLIGHT_RECOVERY_INTERVAL = 0.05  # FASTER: 50ms between recovery attempts
LOWLIGHT_FAST_RECOVERY_COUNT = 5  # More recovery attempts
LOWLIGHT_CONFIRM_FRAMES = 3  # Fewer frames to confirm darkness
LOWLIGHT_RECOVERY_FRAMES = 2  # Fewer frames to confirm recovery

# Police car detection - Based on split red/blue color scheme
POLICE_CONFIRM_FRAMES = 5
POLICE_SEEK_ACCELERATION = 0.70
POLICE_MIN_AREA = 1500
POLICE_MAX_AREA = 60000
POLICE_MIN_RED_RATIO = 0.15
POLICE_MIN_BLUE_RATIO = 0.15
POLICE_SPLIT_RATIO_MIN = 0.30
POLICE_SPLIT_RATIO_MAX = 0.70
POLICE_MIN_RED_PIXELS = 300
POLICE_MIN_BLUE_PIXELS = 300
POLICE_CONFIDENCE_THRESHOLD = 0.65
POLICE_EVENT_TIMEOUT_SECONDS = 5.0  # EV2: 5 seconds to collect red token

# Police car color ranges - Split Red/Blue
POLICE_RED_LOWER1 = np.array([0, 80, 60])
POLICE_RED_UPPER1 = np.array([10, 255, 255])
POLICE_RED_LOWER2 = np.array([170, 80, 60])
POLICE_RED_UPPER2 = np.array([180, 255, 255])
POLICE_BLUE_LOWER = np.array([100, 80, 60])
POLICE_BLUE_UPPER = np.array([130, 255, 255])

# Trailing/Chasing car detection - Based on Cyan/Teal color scheme
TRAILING_CONFIRM_FRAMES = 4
TRAILING_MOTION_AREA_MIN = 3000
TRAILING_APPROACH_RATIO = 1.05
TRAILING_ASPECT_RATIO_MIN = 0.3
TRAILING_ASPECT_RATIO_MAX = 3.0
TRAILING_MIN_HEIGHT_RATIO = 0.08

# Trailing car color ranges - Cyan/Teal
TRAILING_CYAN_LOWER1 = np.array([80, 70, 70])
TRAILING_CYAN_UPPER1 = np.array([110, 255, 255])
TRAILING_CYAN_LOWER2 = np.array([75, 60, 100])
TRAILING_CYAN_UPPER2 = np.array([105, 255, 255])

LANE_FOLLOW_SMOOTHING_PREVIOUS = 0.70
LANE_FOLLOW_SMOOTHING_CURRENT = 0.30
LANE_FOLLOW_MIN_ABS_SLOPE = 0.35
LANE_FOLLOW_MAX_ABS_SLOPE = 3.5
LANE_FOLLOW_MIN_LINES_PER_SIDE = 1
LANE_FOLLOW_INTERVAL_SECONDS = 0.10
LANE_FOLLOW_PROCESS_WIDTH = 320
BACK_DETECTION_INTERVAL_SECONDS = 0.05
PROCESSING_PERIOD_SECONDS = 0.02

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
        'police_lane': None,
    },
    # Front detection
    'police_front_detected': False,
    'police_front_lane': None,
    'police_front_confidence': 0.0,
    'police_front_bbox': None,
    'chasing_front_detected': False,
    'chasing_front_lane': None,
    'chasing_front_confidence': 0.0,
    'chasing_front_bbox': None,
    'front_police_evade': False,
    'front_police_evade_until': 0.0,
    'front_chasing_evade': False,
    'front_chasing_evade_until': 0.0,
    'green_target_debug': None,
    'green_decision_debug': None,
    'police_debug': None,
    'hazard_debug': None,
    'speed_modifier': 1.0,
    'selected_target_type': 'NONE',
    'current_mode': 'LANE FOLLOW',
    # Light detection confirmation
    'light_confirm_count': 0,
    'recovery_confirm_count': 0,
    'last_recovery_burst_time': 0.0,
    # EV tracking
    'ev1_darkness_active': False,
    'ev1_darkness_brake_sent': False,
    'ev2_police_active': False,
    'ev2_police_start_time': 0.0,
    'ev2_red_token_collected': False,
    'ev2_police_penalty': False,
    'ev3_chasing1_active': False,
    'ev3_chasing1_start_time': 0.0,
    'ev3_chasing1_evaded': False,
    'ev4_chasing2_active': False,
    'ev4_chasing2_start_time': 0.0,
    'ev4_chasing2_evaded': False,
    'ev5_golden_lane_active': False,
    'ev5_golden_lane_start_time': 0.0,
    'ev5_golden_lane_target': 0,
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
        cv2.inRange(hsv, np.array((0, 20, 145), dtype=np.uint8), np.array((20, 200, 255), dtype=np.uint8)),
        cv2.inRange(hsv, np.array((160, 20, 145), dtype=np.uint8), np.array((179, 200, 255), dtype=np.uint8))
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
    detected_tokens = []

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

    hsv_bright = cv2.cvtColor(process_roi, cv2.COLOR_BGR2HSV)
    standard_kernel = np.ones((3, 3), np.uint8)

    standard_color_ranges = {
        "green": [((42, 45, 95), (88, 255, 255))],
        "yellow": [((16, 70, 105), (38, 255, 255))]
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
            min_circularity = 0.46 if color_name == "green" else 0.58
            if circularity < min_circularity:
                continue

            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center_x = x * scale
            center_y = y * scale + roi_y1
            radius = radius * scale
            if radius <= 0:
                continue
            if center_y < frame_h * 0.20 or center_y > frame_h * 0.92:
                continue

            x, y, width, height = cv2.boundingRect(cnt)
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

    hsv_original = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        if circularity < 0.46 or circle_fill_ratio < 0.43:
            continue

        if center_y < frame_h * 0.20 or center_y > frame_h * 0.92:
            continue

        detected_tokens.append({
            "color": "red",
            "x": int(center_x),
            "y": int(center_y),
            "radius": int(radius),
            "area": float(area),
            "bbox": (
                max(0, min(frame_w - 1, int(x))),
                max(0, min(frame_h - 1, int(y))),
                max(0, min(frame_w - 1, int(x + width))),
                max(0, min(frame_h - 1, int(y + height)))
            ),
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
        target_center = (
            max(0, min(frame_w - 1, target_x)),
            max(0, min(frame_h - 1, target_y))
        )
        predicted_x = int(green_target_debug.get('predicted_x', target_center[0]))

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

def make_lane_boundary(line_params, y_bottom, y_top):
    slope, intercept = line_params
    if abs(slope) < 0.001:
        return None

    x_bottom = int((y_bottom - intercept) / slope)
    x_top = int((y_top - intercept) / slope)
    return (x_bottom, y_bottom, x_top, y_top)

def detect_lane_following(frame, previous_steering=0.0):
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

# =========================================================
# POLICE CAR DETECTION - Split Red/Blue Color Scheme
# =========================================================
def detect_police_car(frame):
    """
    Detect police car using split red/blue color scheme.
    Police car = vehicle with both vibrant red and deep blue next to each other.
    """
    if frame is None:
        return False, 0.0, None, None

    frame_h, frame_w = frame.shape[:2]

    # Focus on where cars appear
    roi_x1 = int(frame_w * 0.10)
    roi_x2 = int(frame_w * 0.90)
    roi_y1 = int(frame_h * 0.25)
    roi_y2 = int(frame_h * 0.95)

    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_h, roi_w = roi.shape[:2]
    
    if roi_h < 50 or roi_w < 50:
        return False, 0.0, None, None
    
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Red masks
    red_mask1 = cv2.inRange(hsv_roi, POLICE_RED_LOWER1, POLICE_RED_UPPER1)
    red_mask2 = cv2.inRange(hsv_roi, POLICE_RED_LOWER2, POLICE_RED_UPPER2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Blue mask
    blue_mask = cv2.inRange(hsv_roi, POLICE_BLUE_LOWER, POLICE_BLUE_UPPER)
    
    # Clean up
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    
    # Count total pixels
    total_red_pixels = cv2.countNonZero(red_mask)
    total_blue_pixels = cv2.countNonZero(blue_mask)
    
    # Require significant amounts of both colors
    if total_red_pixels < 300 or total_blue_pixels < 300:
        return False, 0.0, None, None
    
    # Find red regions
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_regions = []
    for cnt in red_contours:
        area = cv2.contourArea(cnt)
        if area > 200:
            x, y, w, h = cv2.boundingRect(cnt)
            if 0.3 <= w / max(1, h) <= 4.0:
                red_regions.append((x, y, w, h, area))
    
    # Find blue regions
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_regions = []
    for cnt in blue_contours:
        area = cv2.contourArea(cnt)
        if area > 200:
            x, y, w, h = cv2.boundingRect(cnt)
            if 0.3 <= w / max(1, h) <= 4.0:
                blue_regions.append((x, y, w, h, area))
    
    if len(red_regions) < 1 or len(blue_regions) < 1:
        return False, 0.0, None, None
    
    # Find red and blue regions close to each other
    best_confidence = 0.0
    best_bbox = None
    police_lane = None
    
    for rx, ry, rw, rh, rarea in red_regions:
        for bx, by, bw, bh, barea in blue_regions:
            # Check proximity
            center_red_x = rx + rw/2
            center_red_y = ry + rh/2
            center_blue_x = bx + bw/2
            center_blue_y = by + bh/2
            
            dist = np.sqrt((center_red_x - center_blue_x)**2 + (center_red_y - center_blue_y)**2)
            if dist > 250:
                continue
            
            # Combine bounding boxes
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
            if aspect_ratio < 0.5 or aspect_ratio > 3.0:
                continue
            
            # Verify split colors in combined region
            combined_roi = roi[y1:y2, x1:x2]
            if combined_roi.size == 0 or combined_roi.shape[0] < 20 or combined_roi.shape[1] < 20:
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
            
            # Both colors must be present
            if red_ratio < 0.08 or blue_ratio < 0.08:
                continue
            
            # Split should be balanced (30-70%)
            split_ratio = red_pixels / (red_pixels + blue_pixels) if (red_pixels + blue_pixels) > 0 else 0.5
            if split_ratio < 0.25 or split_ratio > 0.75:
                continue
            
            # Calculate confidence
            confidence = 0.0
            color_presence = min(1.0, (red_pixels / 300 + blue_pixels / 300) / 2.0)
            confidence += color_presence * 0.40
            balance = 1.0 - abs(split_ratio - 0.5) * 2.0
            confidence += balance * 0.30
            area_factor = min(1.0, combined_area / 5000.0)
            confidence += area_factor * 0.30
            
            if confidence > best_confidence and confidence >= 0.60:
                best_confidence = confidence
                center_x = x1 + combined_w/2
                center_y = y1 + combined_h/2
                police_lane = lane_from_x(center_x + roi_x1, center_y + roi_y1, frame_width=frame_w, frame_height=frame_h)
                best_bbox = (x1 + roi_x1, y1 + roi_y1, combined_w, combined_h)
    
    if best_confidence >= 0.60 and best_bbox is not None:
        return True, best_confidence, best_bbox, police_lane
    
    return False, best_confidence, None, None

# =========================================================
# TRAILING/CHASING CAR DETECTION - Cyan/Teal Color Scheme
# =========================================================
def detect_trailing_car(frame):
    """
    Detect trailing/chasing car using cyan/teal color scheme.
    Trailing car = monochromatic cyan/teal vehicle.
    """
    if frame is None:
        return False, 0.0, None, None

    frame_h, frame_w = frame.shape[:2]

    roi_x1 = int(frame_w * 0.10)
    roi_x2 = int(frame_w * 0.90)
    roi_y1 = int(frame_h * 0.25)
    roi_y2 = int(frame_h * 0.95)

    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_h, roi_w = roi.shape[:2]
    
    if roi_h < 50 or roi_w < 50:
        return False, 0.0, None, None
    
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Cyan/Teal masks - multiple ranges for better detection
    cyan_mask1 = cv2.inRange(hsv_roi, TRAILING_CYAN_LOWER1, TRAILING_CYAN_UPPER1)
    cyan_mask2 = cv2.inRange(hsv_roi, TRAILING_CYAN_LOWER2, TRAILING_CYAN_UPPER2)
    cyan_mask = cv2.bitwise_or(cyan_mask1, cyan_mask2)
    
    # Clean up
    kernel = np.ones((5, 5), np.uint8)
    cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_OPEN, kernel)
    cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    trailing = False
    largest_area = 0.0
    largest_bbox = None
    trailing_lane = None
    trailing_confidence = 0.0
    
    with data_lock:
        prev_area = shared_data.get('back_prev_area', 0.0)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < TRAILING_MOTION_AREA_MIN:
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        
        if w < 20 or h < 20 or w > roi_w * 0.8 or h > roi_h * 0.8:
            continue
            
        aspect_ratio = float(w) / h if h > 0 else 0.0
        height_ratio = float(h) / roi_h
        
        if (TRAILING_ASPECT_RATIO_MIN <= aspect_ratio <= TRAILING_ASPECT_RATIO_MAX and 
            height_ratio > TRAILING_MIN_HEIGHT_RATIO and height_ratio < 0.85):
            
            roi_region = roi[y:y+h, x:x+w]
            if roi_region.size > 0 and roi_region.shape[0] > 10 and roi_region.shape[1] > 10:
                hsv_region = cv2.cvtColor(roi_region, cv2.COLOR_BGR2HSV)
                cyan_pixels = cv2.countNonZero(cv2.inRange(hsv_region, TRAILING_CYAN_LOWER1, TRAILING_CYAN_UPPER1))
                cyan_pixels += cv2.countNonZero(cv2.inRange(hsv_region, TRAILING_CYAN_LOWER2, TRAILING_CYAN_UPPER2))
                
                region_area = roi_region.shape[0] * roi_region.shape[1]
                color_purity = cyan_pixels / region_area if region_area > 0 else 0
                
                if color_purity < 0.30:
                    continue
                
                # Confidence based on area and color purity
                area_confidence = min(1.0, area / TRAILING_MOTION_AREA_MIN)
                purity_confidence = min(1.0, color_purity * 2.0)
                confidence = (area_confidence * 0.4 + purity_confidence * 0.6)
                
                # Motion detection
                if prev_area > 0 and area > (prev_area * 1.05):
                    confidence = min(1.0, confidence * 1.2)
                
                if area > largest_area and confidence >= 0.5:
                    largest_area = area
                    center_x = x + w/2
                    center_y = y + h/2
                    trailing_lane = lane_from_x(center_x + roi_x1, center_y + roi_y1, frame_width=frame_w, frame_height=frame_h)
                    largest_bbox = (x + roi_x1, y + roi_y1, w, h)
                    trailing_confidence = confidence
                    
                    if confidence >= 0.6:
                        trailing = True
    
    with data_lock:
        if largest_area > 0:
            shared_data['back_prev_area'] = largest_area
    
    if trailing and trailing_confidence >= 0.5 and largest_bbox is not None:
        return True, largest_area, largest_bbox, trailing_lane
    
    return False, largest_area, largest_bbox, trailing_lane

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
    lane_width = frame_width / 5.0
    return (lane + 2) * lane_width + lane_width / 2.0

def get_safe_lane(current_lane, hazard_lanes, frame_width, bonus_lanes=None):
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
    return None

def is_reachable_green_token(token, frame_width, frame_height, current_lane, blocked_lanes):
    return green_rejection_reason(token, frame_width, frame_height, current_lane, blocked_lanes) is None

def green_target_score(token, frame_width, frame_height, current_lane):
    token_lane = lane_from_x(token['x'], token['y'], frame_width=frame_width)
    center_error = abs(token['x'] - (frame_width / 2.0)) / (frame_width / 2.0)
    lane_center_error = abs(token['x'] - get_lane_center_x(token_lane, frame_width)) / (frame_width / 5.0)
    lane_change_cost = abs(token_lane - current_lane) * 0.01
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

        true_color = token.get('true_color', token.get('color'))
        if true_color not in ['green', 'yellow', 'red']:
            true_color = detect_token_color_robust(front_frame, token['x'], token['y'], token['radius'])
        
        if true_color not in ['green', 'yellow', 'red']:
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
        
        # EV2: Check if red token collected during police event
        if color == 'red' and shared_data.get('ev2_police_active', False):
            shared_data['ev2_red_token_collected'] = True
            print(f"[EV2] Red token collected! Police threat neutralized.")

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
        direction = 2.5 if committed_target_lane > current_lane else -2.5
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
            hsv_tokens = detect_colored_tokens(processed_front_frame)
            
            tokens = merge_token_detections(hsv_tokens)
            
            # =========================================================
            # DETECT FRONT THREATS - POLICE AND CHASING CARS
            # =========================================================
            # Detect police car in front
            police_front, police_front_conf, police_front_bbox, police_front_lane = detect_police_car(front_frame)
            
            # Detect chasing/trailing car in front
            chasing_front, chasing_front_conf, chasing_front_bbox, chasing_front_lane = detect_trailing_car(front_frame)
            
            with data_lock:
                # Store front detections
                shared_data['police_front_detected'] = police_front and police_front_conf >= 0.60
                shared_data['police_front_lane'] = police_front_lane
                shared_data['police_front_confidence'] = police_front_conf
                shared_data['police_front_bbox'] = police_front_bbox
                
                shared_data['chasing_front_detected'] = chasing_front
                shared_data['chasing_front_lane'] = chasing_front_lane
                shared_data['chasing_front_confidence'] = chasing_front_conf
                shared_data['chasing_front_bbox'] = chasing_front_bbox
                
                # EV2: Police car detection - require consistent detection
                police_history = shared_data.get('police_detection_history', [])
                if police_front and police_front_conf >= 0.60:
                    police_history.append(1)
                    if len(police_history) > 5:
                        police_history.pop(0)
                    if sum(police_history) >= 3:
                        if not shared_data.get('ev2_police_active', False):
                            shared_data['ev2_police_active'] = True
                            shared_data['ev2_police_start_time'] = time.time()
                            shared_data['ev2_red_token_collected'] = False
                            shared_data['ev2_police_penalty'] = False
                            print(f"[EV2] 🚨 POLICE CAR CONFIRMED! Collect red token within 5 seconds!")
                else:
                    police_history.append(0)
                    if len(police_history) > 5:
                        police_history.pop(0)
                shared_data['police_detection_history'] = police_history
                
                # EV3 & EV4: Chasing car detection
                if chasing_front and chasing_front_conf >= 0.5:
                    trailing_streak = shared_data.get('trailing_streak', 0) + 1
                    shared_data['trailing_streak'] = trailing_streak
                    shared_data['trailing_lane'] = chasing_front_lane
                    
                    if trailing_streak >= 3:
                        shared_data['trailing_detected'] = True
                        shared_data['danger_detected'] = True
                        
                        if not shared_data.get('ev3_chasing1_active', False) and not shared_data.get('ev4_chasing2_active', False):
                            shared_data['ev3_chasing1_active'] = True
                            shared_data['ev3_chasing1_start_time'] = time.time()
                            shared_data['ev3_chasing1_evaded'] = False
                            print(f"[EV3] 🏎️ CHASING CAR 1 DETECTED! (10 sec to avoid)")
                        elif shared_data.get('ev3_chasing1_active', False) and not shared_data.get('ev4_chasing2_active', False):
                            shared_data['ev4_chasing2_active'] = True
                            shared_data['ev4_chasing2_start_time'] = time.time()
                            shared_data['ev4_chasing2_evaded'] = False
                            print(f"[EV4] 🏎️ CHASING CAR 2 DETECTED! (3 sec to avoid)")
                else:
                    trailing_streak = max(0, shared_data.get('trailing_streak', 0) - 1)
                    shared_data['trailing_streak'] = trailing_streak
                    if trailing_streak < 3:
                        shared_data['trailing_detected'] = False
            
            # If police car detected in front and in same lane, EVADE IMMEDIATELY!
            if police_front and police_front_conf >= 0.60 and police_front_lane == current_lane:
                print(f"[EV2] ⚠️ Police car in lane {police_front_lane}! EVADING IMMEDIATELY!")
                with data_lock:
                    shared_data['front_police_evade'] = True
                    shared_data['front_police_evade_until'] = time.time() + 3.0
            
            # If chasing car detected in front and in same lane, EVADE IMMEDIATELY!
            if chasing_front and chasing_front_conf >= 0.5 and chasing_front_lane == current_lane:
                ev_type = "EV3" if shared_data.get('ev3_chasing1_active', False) and not shared_data.get('ev4_chasing2_active', False) else "EV4"
                print(f"[{ev_type}] ⚠️ Chasing car in lane {chasing_front_lane}! EVADING IMMEDIATELY!")
                with data_lock:
                    shared_data['front_chasing_evade'] = True
                    shared_data['front_chasing_evade_until'] = time.time() + 2.0
            
            # =========================================================
            # SUPER FAST LIGHT DETECTION & RECOVERY - ENHANCED
            # =========================================================
            try:
                gray_full = cv2.cvtColor(front_frame, cv2.COLOR_BGR2GRAY)
                current_brightness = float(np.mean(gray_full))
                
                with data_lock:
                    baseline = shared_data.get('brightness_baseline')
                    low_light_active = shared_data.get('low_light_active', False)
                    light_confirm_count = shared_data.get('light_confirm_count', 0)
                    recovery_confirm_count = shared_data.get('recovery_confirm_count', 0)
                    
                    # Initialize baseline
                    if baseline is None or baseline < 10.0:
                        baseline = max(current_brightness, 50.0)
                        shared_data['brightness_baseline'] = baseline
                        print(f"[LIGHT] Baseline brightness set to: {baseline:.1f}")
                    
                    # Check for split screen (not low light)
                    std_dev = np.std(gray_full)
                    is_split_screen = std_dev > 80
                    
                    # Determine if dark
                    is_dark = False
                    if not is_split_screen and baseline >= LOWLIGHT_MIN_BASELINE:
                        if current_brightness < baseline * LOWLIGHT_DARK_THRESHOLD:
                            is_dark = True
                            if not low_light_active:
                                print(f"[LIGHT] ⚡ DARKNESS DETECTED! Brightness: {current_brightness:.1f} (baseline: {baseline:.1f})")
                                shared_data['recovery_phase'] = 1
                        elif low_light_active and current_brightness < baseline * LOWLIGHT_RECOVERY_THRESHOLD:
                            is_dark = True
                    
                    # Handle darkness detection with confirmation
                    if is_dark and not low_light_active:
                        light_confirm_count += 1
                        if light_confirm_count >= LOWLIGHT_CONFIRM_FRAMES:
                            shared_data['low_light_active'] = True
                            shared_data['low_light_recovery_sent'] = False
                            shared_data['low_light_recovery_start_time'] = time.time()
                            shared_data['recovery_attempts'] = 0
                            shared_data['recovery_completed'] = False
                            shared_data['ev1_darkness_active'] = True
                            shared_data['ev1_darkness_brake_sent'] = False
                            print(f"[EV1] 🌑 DARKNESS CONFIRMED! Must brake/decelerate fully!")
                            light_confirm_count = 0
                    elif not is_dark and low_light_active:
                        recovery_confirm_count += 1
                        if recovery_confirm_count >= LOWLIGHT_RECOVERY_FRAMES:
                            shared_data['low_light_active'] = False
                            shared_data['low_light_recovered'] = True
                            shared_data['lights_on'] = True
                            shared_data['low_light_recovery_sent'] = False
                            shared_data['recovery_completed'] = True
                            shared_data['recovery_phase'] = 0
                            shared_data['run_summary']['light_recovered'] = True
                            shared_data['ev1_darkness_active'] = False
                            recovery_confirm_count = 0
                            print(f"[LIGHT] ✅ LIGHT RESTORED! Brightness: {current_brightness:.1f}")
                            print(f"[EV1] ✅ Darkness event resolved!")
                    else:
                        # Reset confirm counts if conditions change
                        if is_dark:
                            light_confirm_count = min(light_confirm_count + 1, LOWLIGHT_CONFIRM_FRAMES)
                            recovery_confirm_count = 0
                        else:
                            recovery_confirm_count = min(recovery_confirm_count + 1, LOWLIGHT_RECOVERY_FRAMES)
                            light_confirm_count = 0
                    
                    # Update baseline with adaptive learning
                    if not low_light_active:
                        baseline = baseline * 0.95 + current_brightness * 0.05
                    else:
                        baseline = baseline * 0.99 + current_brightness * 0.01
                    
                    shared_data['brightness_baseline'] = max(baseline, 30.0)
                    shared_data['brightness_last'] = current_brightness
                    shared_data['lights_on'] = not low_light_active
                    shared_data['light_confirm_count'] = light_confirm_count
                    shared_data['recovery_confirm_count'] = recovery_confirm_count
                    
                    # Track event timing - Challenge 1: Light event must happen within first 10 seconds
                    elapsed = time.time() - run_start_time if run_start_time else 999.0
                    if low_light_active and elapsed <= LOWLIGHT_EVENT_WINDOW_SECONDS:
                        if not shared_data.get('light_event_triggered', False):
                            shared_data['light_event_triggered'] = True
                            print(f"[LIGHT] 🌑 CHALLENGE 1: Light event triggered at {elapsed:.2f}s")
                            print(f"[LIGHT] ⚡ Send acceleration_input = -1.0 to recover light!")
                
                # Update tokens for low light - All tokens become Yellow
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
            
            # Draw front detections on debug frame
            if police_front and police_front_conf >= 0.60 and police_front_bbox is not None:
                x, y, w, h = police_front_bbox
                cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(debug_frame, f"POLICE FRONT ({police_front_lane})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if chasing_front and chasing_front_conf >= 0.5 and chasing_front_bbox is not None:
                x, y, w, h = chasing_front_bbox
                cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 255), 3)
                cv2.putText(debug_frame, f"CHASING FRONT ({chasing_front_lane})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            with data_lock:
                if shared_data.get('low_light_active', False):
                    cv2.putText(debug_frame, "⚡ EV1: LOW LIGHT - BRAKE FULLY ⚡", (10, 100),
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
                
                ev2_active = shared_data.get('ev2_police_active', False)
                if ev2_active:
                    elapsed = time.time() - shared_data.get('ev2_police_start_time', 0)
                    remaining = max(0, 5.0 - elapsed)
                    cv2.putText(debug_frame, f"🚨 EV2: POLICE ACTIVE - {remaining:.1f}s to collect RED token 🚨", (10, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                ev3_active = shared_data.get('ev3_chasing1_active', False)
                if ev3_active:
                    elapsed = time.time() - shared_data.get('ev3_chasing1_start_time', 0)
                    remaining = max(0, 10.0 - elapsed)
                    cv2.putText(debug_frame, f"🏎️ EV3: CHASING CAR 1 - {remaining:.1f}s to evade", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                ev4_active = shared_data.get('ev4_chasing2_active', False)
                if ev4_active:
                    elapsed = time.time() - shared_data.get('ev4_chasing2_start_time', 0)
                    remaining = max(0, 3.0 - elapsed)
                    cv2.putText(debug_frame, f"🏎️ EV4: CHASING CAR 2 - {remaining:.1f}s to evade", (10, 200),
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
                last_back_detection_time = shared_data.get('last_back_detection_time', 0.0)
                back_debug = dict(shared_data.get('back_detection_debug', {}))

            if now - last_back_detection_time >= BACK_DETECTION_INTERVAL_SECONDS:
                # Use unified detection functions
                police_raw, police_confidence, police_bbox, police_lane = detect_police_car(back_frame)
                trailing_raw, trailing_area, trailing_bbox, trailing_lane = detect_trailing_car(back_frame)
                
                back_debug = {
                    'trailing_raw': trailing_raw,
                    'trailing_area': trailing_area,
                    'trailing_bbox': trailing_bbox,
                    'trailing_lane': trailing_lane,
                    'police_raw': police_raw,
                    'police_confidence': police_confidence,
                    'police_bbox': police_bbox,
                    'police_lane': police_lane,
                }

                with data_lock:
                    shared_data['last_back_detection_time'] = now
                    shared_data['back_detection_debug'] = back_debug

                    # Update trailing detection
                    if trailing_raw:
                        shared_data['trailing_streak'] = shared_data.get('trailing_streak', 0) + 1
                        shared_data['trailing_lane'] = trailing_lane
                    else:
                        shared_data['trailing_streak'] = 0
                    
                    # Update police detection
                    police_history = shared_data.get('police_detection_history', [])
                    police_history.append(1 if police_raw else 0)
                    if len(police_history) > POLICE_CONFIRM_FRAMES:
                        police_history.pop(0)
                    shared_data['police_detection_history'] = police_history
                    
                    police_confirmed = sum(police_history) >= (POLICE_CONFIRM_FRAMES * 0.7)
                    trailing_confirmed = shared_data['trailing_streak'] >= TRAILING_CONFIRM_FRAMES
                    
                    if police_confirmed:
                        shared_data['police_lane'] = police_lane
                    
                    shared_data['trailing_detected'] = trailing_confirmed
                    shared_data['police_detected'] = police_confirmed
                    shared_data['danger_detected'] = trailing_confirmed or police_confirmed
                    shared_data['police_confidence'] = police_confidence if police_raw else shared_data.get('police_confidence', 0.0)
                    
                    if police_confirmed:
                        shared_data['run_summary']['police_appeared'] = True
                        if not shared_data.get('ev2_police_active', False):
                            shared_data['ev2_police_active'] = True
                            shared_data['ev2_police_start_time'] = time.time()
                            shared_data['ev2_red_token_collected'] = False
                            shared_data['ev2_police_penalty'] = False
                            print(f"[EV2] 🚨 POLICE CAR CONFIRMED! Collect red token within 5 seconds!")
                    
                    if trailing_confirmed:
                        shared_data['run_summary']['trailing_appeared'] = True
                        elapsed = time.time() - run_start_time if run_start_time else 0
                        if not shared_data.get('ev3_chasing1_active', False):
                            shared_data['ev3_chasing1_active'] = True
                            shared_data['ev3_chasing1_start_time'] = time.time()
                            shared_data['ev3_chasing1_evaded'] = False
                            print(f"[EV3] 🏎️ CHASING CAR 1 at {elapsed:.2f}s (10 sec to avoid)")
                        elif not shared_data.get('ev4_chasing2_active', False):
                            shared_data['ev4_chasing2_active'] = True
                            shared_data['ev4_chasing2_start_time'] = time.time()
                            shared_data['ev4_chasing2_evaded'] = False
                            print(f"[EV4] 🏎️ CHASING CAR 2 at {elapsed:.2f}s (only 3 sec to avoid)")

            trailing_raw = back_debug.get('trailing_raw', False)
            trailing_area = back_debug.get('trailing_area', 0.0)
            trailing_bbox = back_debug.get('trailing_bbox')
            trailing_lane = back_debug.get('trailing_lane')
            police_raw = back_debug.get('police_raw', False)
            police_confidence = back_debug.get('police_confidence', 0.0)
            police_bbox = back_debug.get('police_bbox')
            police_lane = back_debug.get('police_lane')

            # Check EV2 police time limit
            with data_lock:
                if shared_data.get('ev2_police_active', False):
                    police_elapsed = time.time() - shared_data.get('ev2_police_start_time', 0)
                    if police_elapsed > 5.0:
                        if not shared_data.get('ev2_red_token_collected', False):
                            print(f"[EV2] ❌ Time expired! No red token collected. Speed reduced by 50%!")
                            if not shared_data.get('ev2_police_penalty', False):
                                shared_data['ev2_police_penalty'] = True
                                shared_data['run_summary']['speed_penalties'] = shared_data['run_summary'].get('speed_penalties', 0) + 1
                        shared_data['ev2_police_active'] = False
                
                # Check EV3 time limit
                if shared_data.get('ev3_chasing1_active', False):
                    ev3_elapsed = time.time() - shared_data.get('ev3_chasing1_start_time', 0)
                    if ev3_elapsed > 10.0:
                        if not shared_data.get('ev3_chasing1_evaded', False):
                            print(f"[EV3] ❌ Time expired! Speed reduced by 50%!")
                            if not shared_data.get('trailing_first_penalty_applied', False):
                                shared_data['trailing_first_penalty_applied'] = True
                                shared_data['run_summary']['speed_penalties'] = shared_data['run_summary'].get('speed_penalties', 0) + 1
                        shared_data['ev3_chasing1_active'] = False
                
                # Check EV4 time limit
                if shared_data.get('ev4_chasing2_active', False):
                    ev4_elapsed = time.time() - shared_data.get('ev4_chasing2_start_time', 0)
                    if ev4_elapsed > 3.0:
                        if not shared_data.get('ev4_chasing2_evaded', False):
                            print(f"[EV4] ❌ Time expired! Speed reduced by 50%!")
                            if not shared_data.get('trailing_second_penalty_applied', False):
                                shared_data['trailing_second_penalty_applied'] = True
                                shared_data['run_summary']['speed_penalties'] = shared_data['run_summary'].get('speed_penalties', 0) + 1
                        shared_data['ev4_chasing2_active'] = False

            if not SHOW_BACK_DEBUG:
                dbg = cv2.resize(back_frame.copy(), (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                cv2.putText(dbg, f"Back T:{trailing_raw} P:{police_raw}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                with data_lock:
                    shared_data['display_back_frame'] = dbg.copy()
                return
            
            # Draw back camera frame
            dbg = cv2.resize(back_frame.copy(), (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            scale_x = DISPLAY_WIDTH / back_frame.shape[1]
            scale_y = DISPLAY_HEIGHT / back_frame.shape[0]

            if trailing_bbox is not None and trailing_raw:
                x, y, w, h = trailing_bbox
                cv2.rectangle(
                    dbg,
                    (int(x * scale_x), int(y * scale_y)),
                    (int((x + w) * scale_x), int((y + h) * scale_y)),
                    (0, 255, 255), 3)
                cv2.putText(dbg, f"TRAILING CAR (Lane {trailing_lane})", (int(x * scale_x), int(y * scale_y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if police_bbox is not None and police_raw and police_confidence >= 0.60:
                x, y, w, h = police_bbox
                cv2.rectangle(
                    dbg,
                    (int(x * scale_x), int(y * scale_y)),
                    (int((x + w) * scale_x), int((y + h) * scale_y)),
                    (0, 0, 255), 3)
                cv2.putText(dbg, f"POLICE CAR (Lane {police_lane}) {police_confidence:.2f}", (int(x * scale_x), int(y * scale_y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            status_color = (0, 255, 0)
            if trailing_raw:
                status_color = (0, 255, 255)
            if police_raw and police_confidence >= 0.60:
                status_color = (0, 0, 255)

            lines = [
                "Back Camera",
                f"Trailing: {trailing_raw} (conf: {shared_data['trailing_streak']}) Lane: {trailing_lane}",
                f"Police: {police_raw} (conf: {police_confidence:.2f}) Lane: {police_lane}",
                f"Area: {int(trailing_area)}"
            ]
            
            for idx, line in enumerate(lines):
                cv2.putText(dbg, line, (10, 30 + idx * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
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
    global control_conn, steering_state, tap_loop_count, current_lane, last_token_lane, last_token_time, locked_green_lane, locked_green_until, committed_target_lane, committed_target_until, committed_target_reason, last_token_debug_time, run_start_time, trailing_dodge_lane, trailing_dodge_until, last_trailing_notice, last_police_notice, locked_green_target, locked_green_target_frames, locked_green_target_missing_frames, emergency_evade_lane, emergency_evade_until
    
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
        recovery_phase = shared_data.get('recovery_phase', 0)
        police_lane = shared_data.get('police_lane', None)
        trailing_lane = shared_data.get('trailing_lane', None)
        lane_follow = dict(shared_data.get('lane_follow', {}))
        
        # Front threat detection
        police_front_detected = shared_data.get('police_front_detected', False)
        police_front_lane = shared_data.get('police_front_lane', None)
        chasing_front_detected = shared_data.get('chasing_front_detected', False)
        chasing_front_lane = shared_data.get('chasing_front_lane', None)
        front_police_evade = shared_data.get('front_police_evade', False)
        front_chasing_evade = shared_data.get('front_chasing_evade', False)
        
        # EV tracking
        ev1_darkness_active = shared_data.get('ev1_darkness_active', False)
        ev2_police_active = shared_data.get('ev2_police_active', False)
        ev2_police_start = shared_data.get('ev2_police_start_time', 0)
        ev2_red_collected = shared_data.get('ev2_red_token_collected', False)
        ev3_chasing1_active = shared_data.get('ev3_chasing1_active', False)
        ev3_chasing1_start = shared_data.get('ev3_chasing1_start_time', 0)
        ev4_chasing2_active = shared_data.get('ev4_chasing2_active', False)
        ev4_chasing2_start = shared_data.get('ev4_chasing2_start_time', 0)
        ev5_golden_lane_active = shared_data.get('ev5_golden_lane_active', False)

    steering_input = 0.0
    acceleration_input = CAR_ACCELERATION 
    desired_lane = current_lane
    decision_reason = "MAINTAIN"
    target_debug = "green:none"
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
    
    # =========================================================
    # EV1: DARKNESS - HIGHEST PRIORITY - Brake fully!
    # Challenge 1: Low Light - Send acceleration_input = -1.0
    # =========================================================
    if ev1_darkness_active:
        steering_input = 0.0
        acceleration_input = -1.0  # Full brake as required by Challenge 1
        decision_reason = "EV1_DARKNESS"
        selected_target_type = "EV1_DARKNESS"
        with data_lock:
            shared_data['ev1_darkness_brake_sent'] = True
            shared_data['steering_input'] = steering_input
            shared_data['acceleration_input'] = acceleration_input
            shared_data['speed_modifier'] = speed_modifier
            shared_data['selected_target_type'] = 'EV1_DARKNESS'
            shared_data['current_mode'] = 'EV1_DARKNESS'
        try:
            send_control_packet(steering_input, acceleration_input)
        except Exception as e:
            print(f"Network error: {e}")
            control_conn = None
        return

    # =========================================================
    # SUPER FAST LIGHT RECOVERY - ENHANCED
    # Challenge 1: Send acceleration_input = -1.0 to recover light
    # =========================================================
    if low_light_active and not ev1_darkness_active:
        with data_lock:
            recovery_sent = shared_data.get('low_light_recovery_sent', False)
            last_recovery_time = shared_data.get('last_recovery_time', 0.0)
            recovery_attempts = shared_data.get('recovery_attempts', 0)
            current_brightness = shared_data.get('brightness_last', 0.0)
            baseline = shared_data.get('brightness_baseline', current_brightness)
        
        # Check if light is restored
        if current_brightness > baseline * LOWLIGHT_RECOVERY_THRESHOLD:
            with data_lock:
                shared_data['low_light_active'] = False
                shared_data['lights_on'] = True
                shared_data['low_light_recovery_sent'] = False
                shared_data['recovery_completed'] = True
                shared_data['recovery_phase'] = 0
                shared_data['run_summary']['light_recovered'] = True
                shared_data['ev1_darkness_active'] = False
            print(f"[LIGHT] ✅ LIGHT RESTORED! Brightness: {current_brightness:.1f}")
            print(f"[EV1] ✅ Darkness event resolved!")
            return
        
        time_since_recovery = time.time() - last_recovery_time
        
        # Send recovery burst - FAST!
        if not recovery_sent or time_since_recovery > LOWLIGHT_RECOVERY_INTERVAL:
            steering_input = 0.0
            acceleration_input = -1.0  # Challenge 1: Send acceleration_input = -1.0
            
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
                shared_data['ev1_darkness_active'] = True
            
            print(f"[LIGHT] ⚡ FAST RECOVERY BURST (attempt {recovery_attempts + 1})")
            print(f"[LIGHT] Sending acceleration_input = -1.0 to recover light!")
            
            # Send multiple recovery packets in quick succession
            for i in range(LOWLIGHT_FAST_RECOVERY_COUNT):
                try:
                    send_control_packet(0.0, -1.0)
                    time.sleep(0.02)  # Very fast between packets
                except Exception:
                    pass
            
            # Send an extra burst for good measure
            time.sleep(0.02)
            for i in range(3):
                try:
                    send_control_packet(0.0, -1.0)
                    time.sleep(0.01)
                except Exception:
                    pass
            
            print(f"[LIGHT] ✅ Recovery burst sent! Waiting for light to restore...")
            return
        
        # Continue sending brake signal if still dark
        steering_input = 0.0
        acceleration_input = -1.0  # Keep braking until recovered
        with data_lock:
            shared_data['steering_input'] = steering_input
            shared_data['acceleration_input'] = acceleration_input
            shared_data['speed_modifier'] = speed_modifier
            shared_data['selected_target_type'] = 'LOW_LIGHT'
            shared_data['current_mode'] = 'LOW_LIGHT'
            shared_data['ev1_darkness_active'] = True
        
        try:
            send_control_packet(steering_input, acceleration_input)
        except Exception as e:
            print(f"Network error: {e}")
            control_conn = None
        return

    # =========================================================
    # EV2: POLICE CAR - MUST collect red token within 5 seconds
    # =========================================================
    if ev2_police_active:
        elapsed = time.time() - ev2_police_start
        remaining = max(0, 5.0 - elapsed)
        
        if int(elapsed * 2) != int((elapsed - 0.02) * 2):
            print(f"[EV2] 🚨 POLICE ACTIVE! {remaining:.1f}s to collect RED token!")
        
        red_tokens = [t for t in tokens_snapshot if t.get('color') == 'red' and t['y'] >= TOKEN_DECISION_Y_MIN and t['y'] < img_h * 0.9]
        
        if red_tokens:
            target_red = max(red_tokens, key=lambda t: (t['y'], -abs(t['x'] - (img_w / 2.0))))
            red_lane = lane_from_x(target_red['x'], target_red['y'], frame_width=img_w, frame_height=img_h)
            
            desired_lane = red_lane
            decision_reason = "EV2_RED_TOKEN"
            selected_target_type = "RED"
            acceleration_input = min(acceleration_input, POLICE_SEEK_ACCELERATION)
            
            if police_front_detected and police_front_lane == current_lane:
                safe_lane = get_safe_lane(current_lane, {current_lane, police_front_lane}, img_w)
                if safe_lane != current_lane:
                    desired_lane = safe_lane
                    decision_reason = "EV2_POLICE_EVADE"
                    steering_input = apply_tap_steering(safe_lane)
            
            if target_red['y'] >= img_h * 0.85:
                with data_lock:
                    shared_data['ev2_red_token_collected'] = True
                    shared_data['red_token_collected_during_police'] = True
                    shared_data['ev2_police_active'] = False
                    shared_data['run_summary']['red_collected'] += 1
                decision_reason = "EV2_RED_COLLECTED"
                selected_target_type = "EV2_COMPLETE"
        else:
            if police_front_detected and police_front_lane == current_lane:
                safe_lane = get_safe_lane(current_lane, {current_lane, police_front_lane}, img_w)
                if safe_lane != current_lane:
                    desired_lane = safe_lane
                    decision_reason = "EV2_POLICE_EVADE"
                    steering_input = apply_tap_steering(safe_lane)
        
        if elapsed > 5.0 and not ev2_red_collected:
            with data_lock:
                if not shared_data.get('ev2_police_penalty', False):
                    shared_data['ev2_police_penalty'] = True
                    shared_data['run_summary']['speed_penalties'] = shared_data['run_summary'].get('speed_penalties', 0) + 1
                    speed_penalties += 1
                    speed_modifier = calculate_speed_modifier(green_streak, red_streak, speed_penalties)
            shared_data['ev2_police_active'] = False

    # =========================================================
    # EV3: CHASING CAR 1 - Avoid collision (10 sec)
    # =========================================================
    if ev3_chasing1_active:
        elapsed = time.time() - ev3_chasing1_start
        remaining = max(0, 10.0 - elapsed)
        
        if int(elapsed * 2) != int((elapsed - 0.02) * 2):
            print(f"[EV3] 🏎️ CHASING CAR 1 ACTIVE! {remaining:.1f}s to evade!")
        
        if chasing_front_detected and chasing_front_lane == current_lane:
            safe_lane = get_safe_lane(current_lane, {current_lane, chasing_front_lane}, img_w)
            if safe_lane != current_lane:
                desired_lane = safe_lane
                decision_reason = "EV3_CHASING_EVADE"
                selected_target_type = "EV3_EVADE"
                steering_input = apply_tap_steering(safe_lane)
                with data_lock:
                    shared_data['ev3_chasing1_evaded'] = True
            else:
                acceleration_input = min(acceleration_input, 0.4)
                decision_reason = "EV3_CHASING_BRAKE"
        else:
            with data_lock:
                if not shared_data.get('ev3_chasing1_evaded', False):
                    shared_data['ev3_chasing1_evaded'] = True
                shared_data['ev3_chasing1_active'] = False
        
        if elapsed > 10.0 and not shared_data.get('ev3_chasing1_evaded', False):
            with data_lock:
                if not shared_data.get('trailing_first_penalty_applied', False):
                    shared_data['trailing_first_penalty_applied'] = True
                    shared_data['run_summary']['speed_penalties'] = shared_data['run_summary'].get('speed_penalties', 0) + 1
                    speed_penalties += 1
                    speed_modifier = calculate_speed_modifier(green_streak, red_streak, speed_penalties)
            shared_data['ev3_chasing1_active'] = False

    # =========================================================
    # EV4: CHASING CAR 2 - Avoid collision (3 sec)
    # =========================================================
    if ev4_chasing2_active:
        elapsed = time.time() - ev4_chasing2_start
        remaining = max(0, 3.0 - elapsed)
        
        if int(elapsed * 2) != int((elapsed - 0.02) * 2):
            print(f"[EV4] 🏎️ CHASING CAR 2 ACTIVE! {remaining:.1f}s to evade!")
        
        if chasing_front_detected and chasing_front_lane == current_lane:
            safe_lane = get_safe_lane(current_lane, {current_lane, chasing_front_lane}, img_w)
            if safe_lane != current_lane:
                desired_lane = safe_lane
                decision_reason = "EV4_CHASING_EVADE"
                selected_target_type = "EV4_EVADE"
                steering_input = apply_tap_steering(safe_lane)
                with data_lock:
                    shared_data['ev4_chasing2_evaded'] = True
            else:
                acceleration_input = min(acceleration_input, 0.3)
                decision_reason = "EV4_CHASING_BRAKE"
        else:
            with data_lock:
                if not shared_data.get('ev4_chasing2_evaded', False):
                    shared_data['ev4_chasing2_evaded'] = True
                shared_data['ev4_chasing2_active'] = False
        
        if elapsed > 3.0 and not shared_data.get('ev4_chasing2_evaded', False):
            with data_lock:
                if not shared_data.get('trailing_second_penalty_applied', False):
                    shared_data['trailing_second_penalty_applied'] = True
                    shared_data['run_summary']['speed_penalties'] = shared_data['run_summary'].get('speed_penalties', 0) + 1
                    speed_penalties += 1
                    speed_modifier = calculate_speed_modifier(green_streak, red_streak, speed_penalties)
            shared_data['ev4_chasing2_active'] = False

    # =========================================================
    # EV5: GOLDEN LANE - Be in the golden lane when timer expires
    # =========================================================
    if ev5_golden_lane_active:
        elapsed = time.time() - shared_data.get('ev5_golden_lane_start_time', 0)
        remaining = max(0, 5.0 - elapsed)
        target_lane = shared_data.get('ev5_golden_lane_target', 0)
        
        if int(elapsed * 2) != int((elapsed - 0.02) * 2):
            print(f"[EV5] 🌟 GOLDEN LANE ACTIVE! {remaining:.1f}s to reach lane {target_lane}")
        
        if current_lane != target_lane:
            desired_lane = target_lane
            decision_reason = "EV5_GOLDEN_LANE"
            selected_target_type = "EV5_GOLDEN"
            steering_input = apply_tap_steering(target_lane)
        else:
            with data_lock:
                shared_data['ev5_golden_lane_active'] = False
        
        if elapsed > 5.0:
            if current_lane == target_lane:
                print(f"[EV5] ✅ Success! Was in golden lane {target_lane} at timer expiry!")
            else:
                print(f"[EV5] ❌ Failed! Not in golden lane {target_lane} at timer expiry!")
            with data_lock:
                shared_data['ev5_golden_lane_active'] = False

    # =========================================================
    # FRONT POLICE CAR EVASION (if not in EV)
    # =========================================================
    if (front_police_evade or (police_front_detected and police_front_lane == current_lane)) and not any([ev1_darkness_active, ev2_police_active, ev3_chasing1_active, ev4_chasing2_active]):
        hazard_lanes = {current_lane}
        if police_front_lane is not None:
            hazard_lanes.add(police_front_lane)
        
        safe_lane = get_safe_lane(current_lane, hazard_lanes, img_w)
        
        if safe_lane != current_lane:
            desired_lane = safe_lane
            decision_reason = "FRONT_POLICE_EVADE"
            selected_target_type = "FRONT_POLICE_EVADE"
            acceleration_input = min(acceleration_input, 0.70)
            steering_input = apply_tap_steering(safe_lane)
            with data_lock:
                shared_data['target_lane'] = safe_lane
                shared_data['emergency_evade_lane'] = safe_lane
                shared_data['emergency_evade_until'] = time.time() + 3.0
                shared_data['front_police_evade'] = False
        else:
            acceleration_input = 0.3
            decision_reason = "FRONT_POLICE_BRAKE"

    # =========================================================
    # FRONT CHASING CAR EVASION (if not in EV)
    # =========================================================
    elif (front_chasing_evade or (chasing_front_detected and chasing_front_lane == current_lane)) and not any([ev1_darkness_active, ev2_police_active, ev3_chasing1_active, ev4_chasing2_active]):
        hazard_lanes = {current_lane}
        if chasing_front_lane is not None:
            hazard_lanes.add(chasing_front_lane)
        
        safe_lane = get_safe_lane(current_lane, hazard_lanes, img_w)
        
        if safe_lane != current_lane:
            desired_lane = safe_lane
            decision_reason = "FRONT_CHASING_EVADE"
            selected_target_type = "FRONT_CHASING_EVADE"
            acceleration_input = min(acceleration_input, 0.78)
            steering_input = apply_tap_steering(safe_lane)
            with data_lock:
                shared_data['target_lane'] = safe_lane
                shared_data['emergency_evade_lane'] = safe_lane
                shared_data['emergency_evade_until'] = time.time() + 2.0
                shared_data['front_chasing_evade'] = False
        else:
            acceleration_input = 0.4
            decision_reason = "FRONT_CHASING_BRAKE"

    # =========================================================
    # BACK POLICE CAR EVASION (only if no EV is active)
    # =========================================================
    elif police_active and police_lane is not None and police_lane == current_lane and not any([ev1_darkness_active, ev2_police_active, ev3_chasing1_active, ev4_chasing2_active]):
        safe_lane = get_safe_lane(current_lane, set(), img_w)
        if safe_lane != current_lane:
            desired_lane = safe_lane
            decision_reason = "POLICE_EVADE"
            selected_target_type = "POLICE_EVADE"
            acceleration_input = min(acceleration_input, 0.78)
            steering_input = apply_tap_steering(safe_lane)
            with data_lock:
                shared_data['target_lane'] = safe_lane
                shared_data['emergency_evade_lane'] = safe_lane
                shared_data['emergency_evade_until'] = time.time() + 2.0

    # =========================================================
    # BACK TRAILING CAR EVASION (only if no EV is active)
    # =========================================================
    elif trailing_detected and trailing_lane is not None and trailing_lane == current_lane and not any([ev1_darkness_active, ev2_police_active, ev3_chasing1_active, ev4_chasing2_active]):
        safe_lane = get_safe_lane(current_lane, set(), img_w)
        if safe_lane != current_lane:
            desired_lane = safe_lane
            decision_reason = "TRAILING_EVADE"
            selected_target_type = "TRAILING_EVADE"
            acceleration_input = min(acceleration_input, 0.78)
            steering_input = apply_tap_steering(safe_lane)
            with data_lock:
                shared_data['target_lane'] = safe_lane
                shared_data['emergency_evade_lane'] = safe_lane
                shared_data['emergency_evade_until'] = time.time() + 2.0

    # =========================================================
    # If we're in emergency evade mode, maintain the evade lane
    # =========================================================
    with data_lock:
        if decision_reason in ("FRONT_POLICE_EVADE", "FRONT_CHASING_EVADE", "POLICE_EVADE", "TRAILING_EVADE", "EV3_CHASING_EVADE", "EV4_CHASING_EVADE"):
            if time.time() < shared_data.get('emergency_evade_until', 0):
                steering_input = apply_tap_steering(shared_data.get('emergency_evade_lane', current_lane))
            else:
                shared_data['emergency_evade_lane'] = None
                shared_data['emergency_evade_until'] = 0.0

    # =========================================================
    # POLICE CAR HANDLING - Seek red token (only if no EV is active)
    # =========================================================
    if police_active and decision_reason not in ("FRONT_POLICE_EVADE", "FRONT_CHASING_EVADE", "POLICE_EVADE", "TRAILING_EVADE", "EV1_DARKNESS", "EV2_POLICE_EVADE") and not any([ev1_darkness_active, ev2_police_active, ev3_chasing1_active, ev4_chasing2_active, ev5_golden_lane_active]):
        red_tokens_police = [t for t in tokens_snapshot if t.get('color') == 'red' and t['y'] >= TOKEN_DECISION_Y_MIN]
        police_debug['reason'] = 'waiting_until_timeout'
        
        if police_red_required and red_tokens_police:
            valid_red_tokens = [t for t in red_tokens_police if lane_from_x(t['x'], t['y'], frame_width=img_w, frame_height=img_h) != police_lane]
            if valid_red_tokens:
                target_red = max(valid_red_tokens, key=lambda t: (t['y'], -abs(t['x'] - (img_w / 2.0))))
                red_lane = lane_from_x(target_red['x'], target_red['y'], frame_width=img_w, frame_height=img_h)
                desired_lane = max(-2, min(2, red_lane))
                decision_reason = "POLICE_RED_TOKEN"
                selected_target_type = "RED"
                acceleration_input = min(acceleration_input, POLICE_SEEK_ACCELERATION)
                police_debug['reason'] = 'near_timeout_target_red'
                target_debug = f"police_red:{target_red['x']},{target_red['y']} left:{max(0.0, POLICE_EVENT_TIMEOUT_SECONDS - police_elapsed):.1f}s"
        elif police_elapsed > POLICE_EVENT_TIMEOUT_SECONDS:
            if not red_token_collected:
                with data_lock:
                    if not shared_data.get('police_penalty_applied', False):
                        shared_data['police_penalty_applied'] = True
                        shared_data['run_summary']['speed_penalties'] = shared_data['run_summary'].get('speed_penalties', 0) + 1
                        speed_penalties += 1
                        speed_modifier = calculate_speed_modifier(green_streak, red_streak, speed_penalties)
            shared_data['police_active'] = False
            police_debug['active'] = False
            police_debug['reason'] = 'expired'

    # Avoid red/yellow during normal driving
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
    for t in hazard_tokens:
        t_lane = lane_from_x(t['x'], t['y'], frame_width=img_w, frame_height=img_h)
        hazard_lanes.add(t_lane)
    red_hazard_lanes = set()
    for t in red_hazard_tokens:
        red_hazard_lanes.add(lane_from_x(t['x'], t['y'], frame_width=img_w, frame_height=img_h))
    yellow_hazard_lanes = set()
    for t in yellow_hazard_tokens:
        yellow_hazard_lanes.add(lane_from_x(t['x'], t['y'], frame_width=img_w, frame_height=img_h))

    green_tokens_for_bonus = [t for t in tokens_snapshot if t.get('color') == 'green' and t['y'] >= TOKEN_DECISION_Y_MIN]
    bonus_lanes = set()
    for t in green_tokens_for_bonus:
        t_lane = lane_from_x(t['x'], t['y'], frame_width=img_w, frame_height=img_h)
        bonus_lanes.add(t_lane)
    green_tokens_visible = [t for t in tokens_snapshot if t.get('color') == 'green']
    green_reject_counts = {
        'candidate': len(green_tokens_visible),
        'too_far': 0,
        'outside_roi': 0,
        'blocked_by_hazard': 0,
        'event_override': 0,
        'no_lock': 0,
    }
    for token in green_tokens_visible:
        reason = green_rejection_reason(token, img_w, img_h, current_lane, hazard_lanes)
        if reason:
            green_reject_counts[reason] = green_reject_counts.get(reason, 0) + 1

    if decision_reason in ("MAINTAIN", "POLICE_RED_TOKEN") and (current_lane in red_hazard_lanes or desired_lane in red_hazard_lanes):
        safe_lane = get_safe_lane(current_lane, hazard_lanes, img_w, bonus_lanes)
        if safe_lane != desired_lane:
            if steering_state != 0 and committed_target_lane != safe_lane:
                steering_state = 0
            desired_lane = safe_lane
            decision_reason = "AVOID_RED"
            selected_target_type = "AVOID_RED"
            hazard_debug_tokens = red_hazard_tokens
            acceleration_input = min(acceleration_input, 0.80)
            steering_input = apply_tap_steering(desired_lane)
        elif steering_state != 0:
            decision_reason = "AVOID_RED"
            steering_input = apply_tap_steering(desired_lane)

    if decision_reason in ("MAINTAIN", "POLICE_RED_TOKEN") and (current_lane in yellow_hazard_lanes or desired_lane in yellow_hazard_lanes):
        safe_lane = get_safe_lane(current_lane, hazard_lanes, img_w, bonus_lanes)
        if safe_lane != desired_lane:
            if steering_state != 0 and committed_target_lane != safe_lane:
                steering_state = 0
            desired_lane = safe_lane
            decision_reason = "AVOID_YELLOW"
            selected_target_type = "AVOID_YELLOW"
            hazard_debug_tokens = yellow_hazard_tokens
            acceleration_input = min(acceleration_input, 0.80)
            steering_input = apply_tap_steering(desired_lane)
        elif steering_state != 0:
            decision_reason = "AVOID_YELLOW"
            steering_input = apply_tap_steering(desired_lane)

    # =========================================================
    # TOKEN DECISION - Seek reachable GREEN tokens (only if no EV is active)
    # =========================================================
    if decision_reason == "MAINTAIN" and not any([ev1_darkness_active, ev2_police_active, ev3_chasing1_active, ev4_chasing2_active, ev5_golden_lane_active]):
        selected_green_target, reachable_green_tokens = select_green_target(
            green_tokens_visible,
            None,
            img_w,
            img_h,
            current_lane,
            hazard_lanes
        )

        if selected_green_target is not None:
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
            
            if decision_reason not in ("FRONT_POLICE_EVADE", "FRONT_CHASING_EVADE", "POLICE_EVADE", "TRAILING_EVADE", "EV3_CHASING_EVADE", "EV4_CHASING_EVADE"):
                steering_input = apply_tap_steering(desired_lane)

            target_debug = (
                f"green_target:{selected_green_target['x']},{selected_green_target['y']} "
                f"reach:{len(reachable_green_tokens)}"
            )
        else:
            green_reject_counts['no_lock'] = len(green_tokens_visible)

            # =========================================================
            # GLOBAL FALLBACK POLICY
            # =========================================================
            safe_lane = get_safe_lane(current_lane, hazard_lanes, img_w, bonus_lanes)
            if safe_lane != current_lane:
                desired_lane = safe_lane
                decision_reason = "FALLBACK_AVOID_HAZARD"
                selected_target_type = "FALLBACK"
                acceleration_input = min(acceleration_input, 0.85)
                steering_input = apply_tap_steering(desired_lane)
                target_debug = f"fallback_evade:L{current_lane}->L{safe_lane}"
            else:
                desired_lane = current_lane
                    decision_reason = "FALLBACK_MAINTAIN"
                    selected_target_type = "FALLBACK"
                    if steering_state != 0:
                        steering_input = apply_tap_steering(desired_lane)
                    target_debug = f"fallback_maintain:L{current_lane}"

    if decision_reason not in ("GREEN_TARGET", "PRE_TARGET_GREEN") and locked_green_target_frames > 0:
        locked_green_target_frames = 0
        locked_green_target = None
        locked_green_target_missing_frames = 0

    # =========================================================
    # LANE FOLLOWING (fallback)
    # =========================================================
    if decision_reason in ("MAINTAIN", "FALLBACK_MAINTAIN") and steering_state == 0:
        lane_center_x = lane_follow.get('lane_center_x')
        if lane_follow.get('valid') and lane_center_x is not None:
            steering_input = float(lane_follow.get('smoothed_steering', 0.0))
            decision_reason = "LANE_FOLLOW"
            target_debug = f"lane_center:{lane_center_x:.1f}"

    # Apply speed rules
    acceleration_input = apply_speed_rules(acceleration_input)

    # Mode mapping for display
    mode_map = {
        "EV1_DARKNESS": "🌑 EV1: DARKNESS",
        "EV2_RED_TOKEN": "🚨 EV2: POLICE",
        "EV2_POLICE_EVADE": "🚨 EV2: POLICE EVADE",
        "EV2_RED_COLLECTED": "✅ EV2: COMPLETE",
        "EV3_CHASING_EVADE": "🏎️ EV3: CHASING EVADE",
        "EV3_CHASING_BRAKE": "🏎️ EV3: CHASING BRAKE",
        "EV4_CHASING_EVADE": "🏎️ EV4: CHASING EVADE",
        "EV4_CHASING_BRAKE": "🏎️ EV4: CHASING BRAKE",
        "EV5_GOLDEN_LANE": "🌟 EV5: GOLDEN LANE",
        "FRONT_POLICE_EVADE": "FRONT POLICE MODE",
        "FRONT_CHASING_EVADE": "FRONT CHASING AVOID",
        "POLICE_EVADE": "POLICE MODE",
        "POLICE_RED_TOKEN": "POLICE MODE",
        "TRAILING_EVADE": "TRAILING CAR AVOID",
        "PRE_TARGET_GREEN": "PRE_TARGET_GREEN",
        "GREEN_TARGET": "COLLECT_GREEN",
        "AVOID_RED": "AVOID RED",
        "AVOID_YELLOW": "AVOID YELLOW",
        "FALLBACK_AVOID_HAZARD": "FALLBACK AVOID",
        "FALLBACK_MAINTAIN": "FALLBACK MAINTAIN",
        "LANE_FOLLOW": "LANE FOLLOW"
    }
    
    if ev1_darkness_active:
        current_mode_text = "🌑 EV1: DARKNESS - BRAKE!"
    elif low_light_active:
        current_mode_text = "🌑 LOW LIGHT - RECOVERING..."
    elif ev2_police_active:
        current_mode_text = f"🚨 EV2: POLICE - {max(0, 5.0 - (time.time() - shared_data.get('ev2_police_start_time', 0))):.1f}s"
    elif ev3_chasing1_active:
        current_mode_text = f"🏎️ EV3: CHASING 1 - {max(0, 10.0 - (time.time() - shared_data.get('ev3_chasing1_start_time', 0))):.1f}s"
    elif ev4_chasing2_active:
        current_mode_text = f"🏎️ EV4: CHASING 2 - {max(0, 3.0 - (time.time() - shared_data.get('ev4_chasing2_start_time', 0))):.1f}s"
    elif ev5_golden_lane_active:
        current_mode_text = f"🌟 EV5: GOLDEN LANE - {max(0, 5.0 - (time.time() - shared_data.get('ev5_golden_lane_start_time', 0))):.1f}s"
    elif front_police_evade or (police_front_detected and police_front_lane == current_lane):
        current_mode_text = "🚨 FRONT POLICE EVADE"
    elif front_chasing_evade or (chasing_front_detected and chasing_front_lane == current_lane):
        current_mode_text = "⚠️ FRONT CHASING EVADE"
    elif trailing_detected and decision_reason == "MAINTAIN":
        current_mode_text = "⚠️ TRAILING CAR AVOID"
    elif police_active and decision_reason == "MAINTAIN":
        current_mode_text = "🚨 POLICE ACTIVE"
    else:
        current_mode_text = mode_map.get(decision_reason, "LANE FOLLOW")

    with data_lock:
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
    emergency_evade_lane = None
    emergency_evade_until = 0.0
    
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
        # Light detection confirmation
        shared_data['light_confirm_count'] = 0
        shared_data['recovery_confirm_count'] = 0
        shared_data['last_recovery_burst_time'] = 0.0
        # Front detection initialization
        shared_data['police_front_detected'] = False
        shared_data['police_front_lane'] = None
        shared_data['police_front_confidence'] = 0.0
        shared_data['police_front_bbox'] = None
        shared_data['chasing_front_detected'] = False
        shared_data['chasing_front_lane'] = None
        shared_data['chasing_front_confidence'] = 0.0
        shared_data['chasing_front_bbox'] = None
        shared_data['front_police_evade'] = False
        shared_data['front_police_evade_until'] = 0.0
        shared_data['front_chasing_evade'] = False
        shared_data['front_chasing_evade_until'] = 0.0
        # EV initialization
        shared_data['ev1_darkness_active'] = False
        shared_data['ev1_darkness_brake_sent'] = False
        shared_data['ev2_police_active'] = False
        shared_data['ev2_police_start_time'] = 0.0
        shared_data['ev2_red_token_collected'] = False
        shared_data['ev2_police_penalty'] = False
        shared_data['ev3_chasing1_active'] = False
        shared_data['ev3_chasing1_start_time'] = 0.0
        shared_data['ev3_chasing1_evaded'] = False
        shared_data['ev4_chasing2_active'] = False
        shared_data['ev4_chasing2_start_time'] = 0.0
        shared_data['ev4_chasing2_evaded'] = False
        shared_data['ev5_golden_lane_active'] = False
        shared_data['ev5_golden_lane_start_time'] = 0.0
        shared_data['ev5_golden_lane_target'] = 0
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
            'police_lane': None,
        }
        shared_data['green_target_debug'] = None
        shared_data['green_decision_debug'] = None
        shared_data['police_debug'] = None
        shared_data['hazard_debug'] = None
        shared_data['speed_modifier'] = 1.0
        shared_data['selected_target_type'] = 'NONE'
        shared_data['current_mode'] = 'LANE FOLLOW'
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
    print("\nEVENT PRIORITIES (Highest to Lowest):")
    print("  EV1: 🌑 Darkness - BRAKE FULLY (acceleration_input = -1.0)")
    print("  EV2: 🚨 Police Car - Collect RED token within 5 seconds")
    print("  EV3: 🏎️ Chasing Car 1 - Avoid collision (10 seconds)")
    print("  EV4: 🏎️ Chasing Car 2 - Avoid collision (3 seconds)")
    print("  EV5: 🌟 Golden Lane - Be in golden lane when timer expires")
    print("\nChallenge 1: Low Light")
    print("  - Brightness decreases under 50%")
    print("  - All tokens become Yellow/Unknown")
    print("  - Player must detect brightness change")
    print("  - Send acceleration_input = -1.0 to recover light")
    print("  - Must happen once during first 10 seconds")
    print("\nCar Detection Logic:")
    print("  - Police Car: Split Red/Blue color scheme (50% Red, 50% Blue)")
    print("  - Trailing/Chasing Car: Monochromatic Cyan/Teal color")
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
            'ev1': shared_data.get('ev1_darkness_active', False) or shared_data.get('ev1_darkness_brake_sent', False),
            'ev2': shared_data.get('ev2_police_active', False) or shared_data.get('ev2_red_token_collected', False),
            'ev3': shared_data.get('ev3_chasing1_active', False) or shared_data.get('ev3_chasing1_evaded', False),
            'ev4': shared_data.get('ev4_chasing2_active', False) or shared_data.get('ev4_chasing2_evaded', False),
            'ev5': shared_data.get('ev5_golden_lane_active', False),
            'front_police': shared_data.get('police_front_detected', False),
            'front_chasing': shared_data.get('chasing_front_detected', False)
        }
        light_attempts = shared_data.get('recovery_attempts', 0)
        light_recovered = summary.get('light_recovered', False)
        police_confidence = shared_data.get('police_confidence', 0.0)
        ev2_red_collected = shared_data.get('ev2_red_token_collected', False)
        ev3_evaded = shared_data.get('ev3_chasing1_evaded', False)
        ev4_evaded = shared_data.get('ev4_chasing2_evaded', False)
    
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
        print(f"  EV1: Darkness (Challenge 1): {'✅ COMPLETE' if events['ev1'] else '❌ NOT TRIGGERED'}")
        print(f"  EV2: Police Car:            {'✅ COMPLETE (Red token collected)' if ev2_red_collected else ('⚠️ ACTIVE' if events['ev2'] else '❌ NOT TRIGGERED')}")
        print(f"  EV3: Chasing Car 1:         {'✅ EVADED' if ev3_evaded else ('⚠️ ACTIVE' if events['ev3'] else '❌ NOT TRIGGERED')}")
        print(f"  EV4: Chasing Car 2:         {'✅ EVADED' if ev4_evaded else ('⚠️ ACTIVE' if events['ev4'] else '❌ NOT TRIGGERED')}")
        print(f"  EV5: Golden Lane:           {'✅ COMPLETE' if events['ev5'] else '❌ NOT TRIGGERED'}")
        print(f"  Light Recovery:             {'✅ SUCCESS' if light_recovered else '❌ FAILED'}")
        if light_attempts > 0:
            print(f"    - Recovery attempts: {light_attempts}")
        print(f"  Police Car (Front):         {'✓ DETECTED' if events['front_police'] else '✗ NOT DETECTED'}")
        print(f"  Chasing Car (Front):        {'✓ DETECTED' if events['front_chasing'] else '✗ NOT DETECTED'}")
        
        print("\n" + "-"*60)
        print("EVENTS PASSED:")
        print("-"*60)
        print(f"  🚓 Police Car Appeared:   {'YES' if summary.get('police_appeared') else 'NO'}")
        print(f"  🚘 Trailing Car Appeared: {'YES' if summary.get('trailing_appeared') else 'NO'}")
    
    print("\n" + "="*60 + "\n")