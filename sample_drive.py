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
GREEN_CHASE_ACCELERATION = 0.70
GREEN_STEER_GAIN = 6.0
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
GREEN_TARGET_LOCK_SECONDS = 2.8
YELLOW_EFFECT_DURATION_SECONDS = 5.0
TOKEN_COLLECTION_Y_RATIO = 0.90
TOKEN_COLLECTION_COOLDOWN_SECONDS = 2.0
HAZARD_AVOID_Y_MIN = 25
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
        'light_recovered': False
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
    'evasion_start_time': 0.0
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

def detect_colored_tokens(frame):
    frame_h, frame_w = frame.shape[:2]

    frame_bright = adjust_gamma(frame, gamma=1.5)
    hsv_bright = cv2.cvtColor(frame_bright, cv2.COLOR_BGR2HSV)
    hsv_original = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    detected_tokens = []
    standard_kernel = np.ones((5, 5), np.uint8)

    standard_color_ranges = {
        "green": [((50, 30, 150), (70, 255, 255))],
        "yellow": [((18, 50, 140), (30, 255, 255))]
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
            min_circularity = 0.50 if color_name == "green" else 0.62
            if circularity < min_circularity:
                continue

            (x, y), radius = cv2.minEnclosingCircle(cnt)
            detected_tokens.append({
                "color": color_name,
                "x": int(x),
                "y": int(y),
                "radius": int(radius),
                "area": float(area)
            })

    # Red token detection - AVOID RED TOKENS
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
            "area": float(area)
        })

    return detected_tokens

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
        color = text_colors.get(label, text_colors.get(token["color"], (255, 255, 255)))
        center = (token["x"], token["y"])
        radius = token["radius"]
        cv2.circle(overlay, center, radius, color, -1)
        x1 = max(center[0]-radius, 0)
        y1 = max(center[1]-radius, 0)
        x2 = min(center[0]+radius, frame_w-1)
        y2 = min(center[1]+radius, frame_h-1)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_frame, label.upper(), (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    alpha = 0.3
    cv2.addWeighted(overlay, alpha, display_frame, 1-alpha, 0, display_frame)
    return display_frame

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
    
    # Prefer lanes further from hazards
    best_lane = current_lane
    best_score = -1
    
    for lane in available_lanes:
        # Calculate distance from current lane
        dist = abs(lane - current_lane)
        # Prefer edge lanes when avoiding hazards
        edge_bonus = 1.0 if abs(lane) == 2 else 0.5
        score = dist * edge_bonus
        
        if lane in bonus_lanes:
            score += 100.0  # Massive priority for bonus lanes
        
        if score > best_score:
            best_score = score
            best_lane = lane
            
    return best_lane

def maybe_record_collected_token(tokens, frame_width, frame_height, current_lane, front_frame):
    global locked_green_lane, locked_green_until, last_token_lane, last_token_time, committed_target_lane, committed_target_until, committed_target_reason

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
        lights_on = shared_data.get('lights_on', True)
        if not lights_on:
            print(f"[TOKEN] Light is OFF - green token gives +5% instead of +10%")
        
        locked_green_lane = None
        locked_green_until = 0.0
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
            tokens = detect_colored_tokens(processed_front_frame)
            
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
                
                # When low light active, tokens become unknown (yellow)
                if low_light_active:
                    for token in tokens:
                        token['true_color'] = token['color']
                        token['color'] = 'hidden'
                
                tokens = apply_token_visibility_effects(tokens)
                
            except Exception as e:
                print(f"Brightness detection error: {e}")

            with data_lock:
                shared_data['detected_tokens'] = tokens

            debug_frame = draw_detected_tokens(processed_front_frame, tokens)
            
            with data_lock:
                if shared_data.get('low_light_active', False):
                    cv2.putText(debug_frame, "⚡ LOW LIGHT - FAST RECOVERY ⚡", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                    cv2.putText(debug_frame, "Tokens UNKNOWN (yellow)", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                lights_on = shared_data.get('lights_on', True)
                if not lights_on:
                    cv2.putText(debug_frame, "LIGHTS OFF - Green tokens +5% only", (10, 140),
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
            
            with data_lock:
                decision_debug = shared_data.get('decision_debug', '')
            cv2.putText(debug_frame, f"Tokens: {len(tokens)}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(debug_frame, decision_debug[:95], (10, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
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
            # Enhanced trailing car detection with lane info
            trailing_raw, area, bbox, trailing_lane = detect_trailing_car_enhanced(back_frame)
            
            # Accurate police car detection with lane info
            police_raw, confidence, police_bbox, police_lane = detect_police_car_accurate(back_frame)
            
            with data_lock:
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
                f"Trailing: raw={trailing_raw} confirmed={trailing_confirmed_display} Lane: {display_trailing_lane}",
                f"Police: raw={police_raw} confirmed={police_confirmed_display} conf: {confidence:.2f} Lane: {display_police_lane}",
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
    global control_conn, steering_state, tap_loop_count, current_lane, last_token_lane, last_token_time, locked_green_lane, locked_green_until, committed_target_lane, committed_target_until, committed_target_reason, last_token_debug_time, run_start_time, trailing_dodge_lane, trailing_dodge_until, last_trailing_notice, last_police_notice
    
    if control_conn is None:
        return
    
    with data_lock:
        tokens_snapshot = list(shared_data.get('detected_tokens', []))
        front_frame = shared_data.get('latest_front_frame')
        run_summary = shared_data.get('run_summary', {})
        green_streak = run_summary.get('green_collected', 0)
        red_streak = run_summary.get('red_collected', 0)
        trailing_detected = shared_data.get('trailing_detected', False)
        police_detected = shared_data.get('police_detected', False)
        police_active = shared_data.get('police_active', False)
        low_light_active = shared_data.get('low_light_active', False)
        lights_on = shared_data.get('lights_on', True)
        red_token_collected = shared_data.get('red_token_collected_during_police', False)
        police_lane = shared_data.get('police_lane', None)
        trailing_lane = shared_data.get('trailing_lane', None)

    steering_input = 0.0
    acceleration_input = CAR_ACCELERATION 

    if run_start_time is not None and time.time() - run_start_time < START_CENTER_HOLD_SECONDS:
        try: 
            control_conn.sendall(struct.pack('ff', 0.0, 0.0))
        except Exception: 
            pass
        current_lane = START_LANE
        return

    img_w, img_h = (front_frame.shape[1], front_frame.shape[0]) if front_frame is not None else (640, 480)
    
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

    # =========================================================
    # EMERGENCY EVASION - Police car in same lane (front)
    # =========================================================
    if police_detected and police_lane is not None and police_lane == current_lane:
        print(f"[POLICE] ⚠️ Police car in lane {police_lane}! EVADING!")
        # Find safe lane to move to
        safe_lane = get_safe_lane(current_lane, hazard_lanes, img_w, bonus_lanes)
        if safe_lane != current_lane:
            steering_input = 1.0 if safe_lane > current_lane else -1.0
            acceleration_input = min(acceleration_input, 0.65)
            with data_lock:
                shared_data['target_lane'] = safe_lane
            try:
                send_control_packet(steering_input, acceleration_input)
            except Exception as e:
                print(f"Network error: {e}")
                control_conn = None
            return

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
                    acceleration_input = CAR_ACCELERATION * 0.5
            elif shared_data.get('trailing_second_triggered', False):
                # Second trailing car - only 3 seconds to avoid
                time_since_second = elapsed - shared_data.get('trailing_second_time', 0)
                if time_since_second > 3.0:
                    print(f"[TRAILING] ⚠️ Time expired! Speed reduced by 50%!")
                    acceleration_input = CAR_ACCELERATION * 0.5
        
        # Find safe lane to move to
        safe_lane = get_safe_lane(current_lane, hazard_lanes, img_w, bonus_lanes)
        if safe_lane != current_lane:
            steering_input = 1.0 if safe_lane > current_lane else -1.0
            acceleration_input = min(acceleration_input, 0.65)
            with data_lock:
                shared_data['target_lane'] = safe_lane
            try:
                send_control_packet(steering_input, acceleration_input)
            except Exception as e:
                print(f"Network error: {e}")
                control_conn = None
            return

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
        acceleration_input = MIN_ACCELERATION_WHEN_SLOWED
        
        try:
            send_control_packet(steering_input, acceleration_input)
        except Exception as e:
            print(f"Network error: {e}")
            control_conn = None
        return

    # =========================================================
    # POLICE CAR HANDLING - Seek red token (only when police active)
    # =========================================================
    if police_active:
        police_elapsed = time.time() - shared_data.get('police_start_time', 0)
        
        if police_elapsed > 10.0:
            if not red_token_collected:
                print(f"[POLICE] Time expired! Speed reduced by 50%!")
                acceleration_input = CAR_ACCELERATION * 0.5
            shared_data['police_active'] = False
        else:
            # Seek red tokens (only when police is active)
            red_tokens_police = [t for t in tokens_snapshot if t.get('color') == 'red' and t['y'] >= TOKEN_DECISION_Y_MIN]
            if red_tokens_police:
                target_red = max(red_tokens_police, key=lambda t: t['y'])
                red_lane = lane_from_x(target_red['x'], target_red['y'], frame_width=img_w, frame_height=img_h)
                chosen_target_lane = max(-2, min(2, red_lane))
                print(f"[POLICE] Seeking red token in lane {chosen_target_lane} ({10 - police_elapsed:.1f}s left)")
                
                red_error = (target_red['x'] - (img_w / 2.0)) / (img_w / 2.0)
                if abs(red_error) > TOKEN_CENTER_X_DEADZONE:
                    visual_steer = max(-1.0, min(1.0, red_error * GREEN_STEER_GAIN))
                    if abs(visual_steer) < GREEN_APPROACH_MIN_STEER:
                        visual_steer = GREEN_APPROACH_MIN_STEER if visual_steer > 0 else -GREEN_APPROACH_MIN_STEER
                    steering_input = visual_steer
                else:
                    steering_input = 1.0 if chosen_target_lane > current_lane else -1.0 if chosen_target_lane < current_lane else 0.0
                
                acceleration_input = min(acceleration_input, POLICE_SEEK_ACCELERATION)
                
                try:
                    send_control_packet(steering_input, acceleration_input)
                except Exception as e:
                    print(f"Network error: {e}")
                    control_conn = None
                return
            else:
                # No red tokens visible, just maintain
                acceleration_input = min(acceleration_input, 0.65)

    # =========================================================
    # TOKEN DECISION - Seek GREEN tokens (only when no threats)
    # =========================================================
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
            best_target_lane = max(valid_lanes, key=lambda l: lane_density[l])
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

    # Calculate speed modifiers
    if lights_on:
        estimated_speed_modifier = 1.0 + (green_streak * 0.10) - (red_streak * 0.20)
    else:
        estimated_speed_modifier = 1.0 + (green_streak * 0.05) - (red_streak * 0.20)
    
    estimated_speed_modifier = max(0.70, estimated_speed_modifier)

    acceleration_input = max(
        MIN_ACCELERATION_WHEN_SLOWED,
        min(1.0, acceleration_input)
    )

    with data_lock:
        shared_data['decision_debug'] = f"Police: {police_detected} Lane:{police_lane} | Trailing: {trailing_detected} Lane:{trailing_lane} | YellowHaz:{sorted(list(yellow_hazard_lanes))} | Steer: {steering_input:.2f} | Accel: {acceleration_input:.2f}"

    try:
        send_control_packet(steering_input, acceleration_input)
    except Exception as e:
        print(f"Network error: {e}")
        control_conn = None

# ---------------------------------------------------------
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
        shared_data['police_confidence'] = 0.0
        shared_data['police_detection_history'] = []
        shared_data['police_lane'] = None
        shared_data['trailing_lane'] = None
        shared_data['police_bbox'] = None
        shared_data['trailing_bbox'] = None
        shared_data['police_last_seen_time'] = 0.0
        shared_data['trailing_last_seen_time'] = 0.0
        shared_data['run_summary']['light_recovery_attempts'] = 0
        shared_data['run_summary']['light_recovered'] = False

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
    t_processing = RTTask("Processing", period=0.005, priority=TaskPriority.MEDIUM, execute_func=processing_task)
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