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

# Shared Resources with Mutex Lock for Concurrency
shared_data = {
    'latest_front_frame': None,
    'latest_back_frame': None,
    'steering_input' : 0.0,
    'acceleration_input' : 0.0,
    'detected_tokens': [],
    'target_lane': START_LANE, # Default: Stay in Center Lane (0)
    'danger_detected': False,  # Default: No trailing car danger
    'decision_debug': '',
    'lights_on':       False,  # Default: Lights off
    'police_detected': False,  # Default: No police car
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

# Liz added: shared state for the three challenges
shared_data.update({
    'low_light_active': False,
    'low_light_recovered': False,
    'brightness_baseline': None,
    'brightness_last': 0.0,
    'trailing_streak': 0,
    'police_streak': 0,
    'display_front_frame': None,
    'display_back_frame': None,
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

# Liz added: challenge handling tunables (low light, chasing car, police)
LOWLIGHT_MIN_BASELINE = 35.0
LOWLIGHT_ABSOLUTE_MAX = 50.0
LOWLIGHT_DARK_RATIO = 0.55
LOWLIGHT_RECOVER_RATIO = 0.80
LOWLIGHT_RECOVERY_ACCEL = -1.0
LOWLIGHT_BASELINE_ALPHA = 0.05
LOWLIGHT_BASELINE_MAX_RISE = 1.12
LOWLIGHT_EVENT_WINDOW_SECONDS = 10.0
POLICE_CONFIRM_FRAMES = 6
TRAILING_CONFIRM_FRAMES = 4
TRAILING_DODGE_HOLD_SECONDS = 1.0
POLICE_SEEK_ACCELERATION = 0.64

# region debug instrumentation (remove after verification)
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
    standard_kernel = np.ones((5, 5), np.uint8)

    # -----------------------------------------------------
    # Green and yellow detection
    # -----------------------------------------------------
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
            "x": int(center_x),
            "y": int(center_y),
            "radius": int(radius),
            "area": float(area)
        })

    # Optional tuning window. Uncomment temporarily if you need to inspect
    # which pixels are selected as red:
    # cv2.imshow("Red Token Mask", cv2.resize(red_mask, (640, 480)))
    # cv2.waitKey(1)

    return detected_tokens

def draw_detected_tokens(frame, tokens):
    """
    Draw bounding boxes and semi-transparent overlays for all detected tokens.
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
    for token in tokens:
        label = token.get("display_color", token["color"])
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

    # Liz added: Challenge 3 — police lightbar detection on back camera
    p_roi_x1 = int(frame_w * 0.25)
    p_roi_x2 = int(frame_w * 0.75)
    p_roi_y1 = int(frame_h * 0.45)
    p_roi_y2 = int(frame_h * 0.92)
    
    p_roi = back_frame[p_roi_y1:p_roi_y2, p_roi_x1:p_roi_x2]
    hsv_roi = cv2.cvtColor(p_roi, cv2.COLOR_BGR2HSV)
    roi_h, roi_w = p_roi.shape[:2]
    b_ch, g_ch, r_ch = cv2.split(p_roi)
    gray_p = cv2.cvtColor(p_roi, cv2.COLOR_BGR2GRAY)
    bright_mask = cv2.threshold(gray_p, 175, 255, cv2.THRESH_BINARY)[1]

    lower_red1 = np.array([0, 140, 170])
    upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([168, 140, 170])
    upper_red2 = np.array([180, 255, 255])
    lower_blue = np.array([95, 140, 170])
    upper_blue = np.array([145, 255, 255])

    red_mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    blue_mask = cv2.inRange(hsv_roi, lower_blue, upper_blue)
    red_bgr = cv2.bitwise_and(cv2.inRange(r_ch, 175, 255), bright_mask)
    blue_bgr = cv2.bitwise_and(cv2.inRange(b_ch, 175, 255), bright_mask)
    red_mask = cv2.bitwise_or(red_mask, red_bgr)
    blue_mask = cv2.bitwise_or(blue_mask, blue_bgr)

    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    police = False
    POLICE_MIN_BRIGHT_AREA = 80
    POLICE_PAIR_MAX_X = roi_w * 0.38
    POLICE_PAIR_MAX_Y = roi_h * 0.18
    red_pixels = cv2.countNonZero(red_mask)
    blue_pixels = cv2.countNonZero(blue_mask)

    red_centers = []
    red_areas = []
    for cnt in red_contours:
        area = cv2.contourArea(cnt)
        if area > POLICE_MIN_BRIGHT_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            red_centers.append((x + w/2.0, y + h/2.0))
            red_areas.append(area)

    blue_centers = []
    blue_areas = []
    for cnt in blue_contours:
        area = cv2.contourArea(cnt)
        if area > POLICE_MIN_BRIGHT_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            blue_centers.append((x + w/2.0, y + h/2.0))
            blue_areas.append(area)

    if red_centers and blue_centers and red_pixels > 250 and blue_pixels > 250:
        for rx, ry in red_centers:
            for bx, by in blue_centers:
                if abs(rx - bx) <= POLICE_PAIR_MAX_X and abs(ry - by) <= POLICE_PAIR_MAX_Y:
                    police = True
                    break
            if police:
                break

    if debug_should_log('police_back'):
        debug_session_log('sample_drive.py:detect_trailing_and_police',
                      'back camera police scan',
                      {'police': police,
                       'red_centers': len(red_centers),
                       'blue_centers': len(blue_centers),
                       'red_pixels': int(cv2.countNonZero(red_mask)),
                       'blue_pixels': int(cv2.countNonZero(blue_mask)),
                       'trailing': trailing},
                      'E,F',
                      run_id='verify')

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

        color = token.get('true_color', token.get('color'))
        if color not in ['green', 'yellow', 'red']:
            continue

        if color == 'red':
            token_is_centered = abs(token['x'] - (frame_width / 2.0)) <= frame_width * RED_COLLECTION_CENTER_X_RATIO
            if token_lane != current_lane and not token_is_centered:
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

def processing_task():
    #This is where you write your image processing code to decide how to control the car
    #You can use libraries like OpenCV to process the image
    #There is no limtation to the complexity of the processing task, you can use any libraries you want
    #Remember to use the shared_data to get the latest frame
    with data_lock:
        front_frame = shared_data['latest_front_frame']
        back_frame = shared_data.get('latest_back_frame')
        front_connected = front_camera_sock is not None
        back_connected = back_camera_sock is not None

    if front_frame is not None:
        try:
            processed_front_frame = apply_camera_effects(front_frame)
            tokens = detect_colored_tokens(processed_front_frame)
            tokens = apply_token_visibility_effects(tokens)

            # Liz added: Challenge 1 — low-light detection via front-camera brightness
            try:
                gray_full = cv2.cvtColor(front_frame, cv2.COLOR_BGR2GRAY)
                brightness = float(np.mean(gray_full))
                with data_lock:
                    baseline = shared_data.get('brightness_baseline')
                    was_dark = shared_data.get('low_light_active', False)
                    already_recovered = shared_data.get('low_light_recovered', False)
                    elapsed = (time.time() - run_start_time) if run_start_time else 999.0
                    in_event_window = elapsed <= LOWLIGHT_EVENT_WINDOW_SECONDS
                    if baseline is None:
                        baseline = brightness
                    if baseline > brightness * 1.35 and brightness >= LOWLIGHT_MIN_BASELINE:
                        baseline = brightness
                    candidate_dark = was_dark
                    if brightness >= max(baseline * LOWLIGHT_RECOVER_RATIO, LOWLIGHT_ABSOLUTE_MAX):
                        candidate_dark = False
                    elif (brightness < LOWLIGHT_ABSOLUTE_MAX
                            and baseline >= LOWLIGHT_MIN_BASELINE
                            and brightness < baseline * LOWLIGHT_DARK_RATIO):
                        candidate_dark = True
                    if candidate_dark:
                        if was_dark or (in_event_window and not already_recovered):
                            is_dark = True
                        else:
                            is_dark = False
                    else:
                        is_dark = False
                        if was_dark:
                            shared_data['low_light_recovered'] = True
                    if not is_dark:
                        sample = min(brightness, baseline * LOWLIGHT_BASELINE_MAX_RISE)
                        baseline = (1.0 - LOWLIGHT_BASELINE_ALPHA) * baseline + LOWLIGHT_BASELINE_ALPHA * sample
                    shared_data['brightness_baseline'] = baseline
                    shared_data['brightness_last'] = brightness
                    shared_data['low_light_active'] = is_dark
                    shared_data['lights_on'] = not is_dark
                if is_dark:
                    tokens = []
                if debug_should_log('brightness'):
                    elapsed = (time.time() - run_start_time) if run_start_time else -1.0
                    debug_session_log('sample_drive.py:processing_task',
                                  'front brightness sample',
                                  {'brightness': round(brightness, 2),
                                   'baseline': round(baseline, 2),
                                   'is_dark': is_dark,
                                   'tokens': len(tokens),
                                   'elapsed_s': round(elapsed, 2)},
                                  'A,B',
                                  run_id='verify')
            except Exception:
                pass

            with data_lock:
                shared_data['detected_tokens'] = tokens

            debug_frame = draw_detected_tokens(processed_front_frame, tokens)
            with data_lock:
                if shared_data.get('low_light_active', False):
                    cv2.putText(debug_frame, "LOW LIGHT: tokens unknown", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
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
        except Exception:
            front_status = "Connected - waiting for video stream..." if front_connected else "Not connected yet"
            with data_lock:
                shared_data['display_front_frame'] = make_camera_placeholder("Front Camera", front_status)
    else:
        front_status = "Connected - waiting for video stream..." if front_connected else "Not connected yet"
        with data_lock:
            shared_data['display_front_frame'] = make_camera_placeholder("Front Camera", front_status)

    # Run trailing / police detection on back camera
    if back_frame is not None:
        try:
            trailing_raw, police_raw, area, bbox = detect_trailing_and_police(back_frame)
            with data_lock:
                if trailing_raw:
                    shared_data['trailing_streak'] = shared_data.get('trailing_streak', 0) + 1
                else:
                    shared_data['trailing_streak'] = 0
                if police_raw:
                    shared_data['police_streak'] = shared_data.get('police_streak', 0) + 1
                else:
                    shared_data['police_streak'] = 0
                trailing = shared_data['trailing_streak'] >= TRAILING_CONFIRM_FRAMES
                police = shared_data['police_streak'] >= POLICE_CONFIRM_FRAMES
                shared_data['trailing_detected'] = trailing
                shared_data['police_detected'] = police
                shared_data['danger_detected'] = trailing
                
                if police:
                    shared_data['run_summary']['police_appeared'] = True
                if trailing:
                    shared_data['run_summary']['trailing_appeared'] = True

            dbg = cv2.resize(back_frame.copy(), (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            scale_x = DISPLAY_WIDTH / back_frame.shape[1]
            scale_y = DISPLAY_HEIGHT / back_frame.shape[0]
            if bbox is not None and trailing:
                x, y, w, h = bbox
                cv2.rectangle(
                    dbg,
                    (int(x * scale_x), int(y * scale_y)),
                    (int((x + w) * scale_x), int((y + h) * scale_y)),
                    (0, 255, 255), 2)

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
            with data_lock:
                shared_data['display_back_frame'] = dbg.copy()
        except Exception:
            back_status = "Connected - waiting for video stream..." if back_connected else "Not connected yet"
            with data_lock:
                shared_data['display_back_frame'] = make_camera_placeholder("Back Camera", back_status)
    else:
        back_status = "Connected - waiting for video stream..." if back_connected else "Not connected yet"
        with data_lock:
            shared_data['display_back_frame'] = make_camera_placeholder("Back Camera", back_status)

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
committed_target_lane = None
committed_target_until = 0.0
committed_target_reason = "MAINTAIN"
last_token_debug_time = 0.0
run_start_time = None
trailing_dodge_lane = None
trailing_dodge_until = 0.0
last_trailing_notice = None
last_police_notice = None
def send_controls_task():
    global control_conn, steering_state, tap_loop_count, current_lane, last_token_lane, last_token_time, locked_green_lane, locked_green_until, committed_target_lane, committed_target_until, committed_target_reason, last_token_debug_time, run_start_time, trailing_dodge_lane, trailing_dodge_until, last_trailing_notice, last_police_notice
    
    if control_conn is None:
        return
    
    with data_lock:
        tokens_snapshot   = list(shared_data.get('detected_tokens', []))
        front_frame       = shared_data.get('latest_front_frame')
        run_summary       = shared_data.get('run_summary', {})
        green_streak      = run_summary.get('green_collected', 0)
        red_streak        = run_summary.get('red_collected', 0)
        trailing_detected = shared_data.get('trailing_detected', False)
        police_detected   = shared_data.get('police_detected', False)
        low_light_active  = shared_data.get('low_light_active', False)

    steering_input = 0.0
    acceleration_input = CAR_ACCELERATION 

    if run_start_time is not None and time.time() - run_start_time < START_CENTER_HOLD_SECONDS:
        try: control_conn.sendall(struct.pack('ff', 0.0, 0.0))
        except Exception: pass
        current_lane = START_LANE
        return

    img_w, img_h = (front_frame.shape[1], front_frame.shape[0]) if front_frame is not None else (640, 480)
    maybe_record_collected_token(tokens_snapshot, img_w, img_h)

    # Liz added: Challenge 1 — send accel=-1.0 to restore light while screen is dimmed
    if low_light_active:
        steering_input = 0.0
        acceleration_input = LOWLIGHT_RECOVERY_ACCEL
        if debug_should_log('lowlight_recovery'):
            with data_lock:
                b_last = shared_data.get('brightness_last', 0.0)
                b_base = shared_data.get('brightness_baseline')
            debug_session_log('sample_drive.py:send_controls_task',
                          'LOW LIGHT recovery: sending accel=-1.0',
                          {'brightness': round(b_last, 2),
                           'baseline': round(b_base, 2) if b_base else None,
                           'accel': LOWLIGHT_RECOVERY_ACCEL},
                          'C',
                          run_id='verify')
        try:
            send_control_packet(steering_input, acceleration_input)
        except Exception as e:
            print(f"Network error: {e}")
            control_conn = None
        return

    with data_lock:
        run_summary = shared_data.get('run_summary', {})
        green_streak = run_summary.get('green_collected', 0)
        red_streak = run_summary.get('red_collected', 0)

    # =========================================================
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
    best_yellow = None
    any_green_visible = False
    any_yellow_visible = False
    hazard_lanes = set()
    closest_hazard = None
    green_y_by_lane = {}
    green_target_by_lane = {}
    active_green_target = None
    active_yellow_target = None

    for t in tokens_snapshot:
        color = t.get('color')
        if color not in ['green', 'red', 'yellow'] or t['y'] < TOKEN_DECISION_Y_MIN:
            continue

        t_lane = lane_from_x(t['x'], t['y'], frame_width=img_w, frame_height=img_h)
        t_lane = max(-2, min(2, t_lane))

        if color == 'red' and t['y'] >= HAZARD_AVOID_Y_MIN:
            hazard_lanes.add(t_lane)
            if closest_hazard is None or t['y'] > closest_hazard['y']:
                closest_hazard = {'lane': t_lane, 'x': t['x'], 'y': t['y'], 'color': color}
            continue

        if color == 'yellow':
            any_yellow_visible = True
            lane_gap = abs(t_lane - current_lane)
            candidate = {
                'lane': t_lane,
                'x': t['x'],
                'y': t['y'],
                'score': t['y'] - (lane_gap * img_h * 0.18)
            }
            if lane_gap == 0:
                candidate['score'] += img_h * 0.10

            if best_yellow is None or candidate['score'] > best_yellow['score']:
                best_yellow = candidate
            continue

        if color != 'green':
            continue

        any_green_visible = True
        green_y_by_lane[t_lane] = max(green_y_by_lane.get(t_lane, 0), t['y'])
        lane_green_target = green_target_by_lane.get(t_lane)
        if lane_green_target is None or t['y'] > lane_green_target['y']:
            green_target_by_lane[t_lane] = {'lane': t_lane, 'x': t['x'], 'y': t['y']}
        lane_gap = abs(t_lane - current_lane)
        candidate = {
            'lane': t_lane,
            'x': t['x'],
            'y': t['y'],
            'score': t['y'] - (lane_gap * img_h * 0.15)
        }
        if lane_gap == 0:
            candidate['score'] += img_h * 0.12

        if best_green is None or candidate['score'] > best_green['score']:
            best_green = candidate

    tokens_visible = any_green_visible or any_yellow_visible or bool(hazard_lanes)
    if tokens_visible:
        acceleration_input = min(acceleration_input, TOKEN_VISIBLE_MAX_ACCELERATION)
    elif estimated_speed_modifier > 1.25:
        acceleration_input = min(acceleration_input, HIGH_SPEED_MAX_ACCELERATION)

    # =========================================================
    # 3. GREEN-FIRST TARGET DECISION
    # =========================================================
    chosen_target_lane = None
    decision_reason = "MAINTAIN"
    green_debug = "none"
    yellow_debug = "none"
    green_error = None
    visual_steer = 0.0
    steering_source = "none"

    now = time.time()
    committed_green_active = committed_target_reason in ["COLLECT_GREEN", "GREEN_LOCK", "WAIT_GREEN"]
    committed_yellow_active = committed_target_reason in ["COLLECT_YELLOW", "YELLOW_LOCK"]
    committed_green_visible = (
        committed_green_active and
        committed_target_lane is not None and
        committed_target_lane not in hazard_lanes and
        green_y_by_lane.get(committed_target_lane, 0) > 0
    )
    committed_yellow_visible = (
        committed_yellow_active and
        committed_target_lane is not None and
        committed_target_lane not in hazard_lanes and
        best_yellow is not None and
        best_yellow['lane'] == committed_target_lane
    )

    if best_green is not None and best_green['lane'] not in hazard_lanes:
        active_green_target = best_green
        chosen_target_lane = best_green['lane']
        green_debug = f"lane {best_green['lane']} y {best_green['y']}"
        locked_green_lane = chosen_target_lane
        locked_green_until = now + GREEN_TARGET_LOCK_SECONDS
        decision_reason = "COLLECT_GREEN"
        last_token_lane = chosen_target_lane
        last_token_time = now
    elif committed_green_visible:
        active_green_target = green_target_by_lane.get(committed_target_lane)
        chosen_target_lane = committed_target_lane
        green_debug = f"committed lane {committed_target_lane} y {green_y_by_lane[committed_target_lane]}"
        decision_reason = "GREEN_LOCK"
    elif locked_green_lane is not None and now < locked_green_until and locked_green_lane not in hazard_lanes:
        chosen_target_lane = locked_green_lane
        decision_reason = "GREEN_LOCK"
    elif locked_green_lane is not None and locked_green_lane in hazard_lanes:
        locked_green_lane = None
        locked_green_until = 0.0
        decision_reason = "AVOID_RED_YELLOW"
    elif best_yellow is not None and best_yellow['lane'] not in hazard_lanes:
        active_yellow_target = best_yellow
        chosen_target_lane = best_yellow['lane']
        yellow_debug = f"lane {best_yellow['lane']} y {best_yellow['y']}"
        decision_reason = "COLLECT_YELLOW"
        last_token_lane = chosen_target_lane
        last_token_time = now
    elif committed_yellow_visible:
        active_yellow_target = best_yellow
        chosen_target_lane = committed_target_lane
        yellow_debug = f"committed lane {committed_target_lane} y {best_yellow['y']}"
        decision_reason = "YELLOW_LOCK"

    if not any_green_visible and not any_yellow_visible:
        last_token_lane = None
        last_token_time = 0.0

    token_target_active = decision_reason in ["COLLECT_GREEN", "GREEN_LOCK", "COLLECT_YELLOW", "YELLOW_LOCK"]
    current_lane_blocked = current_lane in hazard_lanes
    target_lane_blocked = chosen_target_lane in hazard_lanes if chosen_target_lane is not None else False
    red_in_current_path = current_lane_blocked and (
        chosen_target_lane is None or
        chosen_target_lane == current_lane or
        decision_reason == "MAINTAIN"
    )

    if target_lane_blocked or red_in_current_path or (current_lane_blocked and not token_target_active):
        clear_lanes = [lane for lane in [-2, -1, 0, 1, 2] if lane not in hazard_lanes]
        if clear_lanes:
            red_lane = closest_hazard['lane'] if closest_hazard is not None else current_lane
            preferred_lanes = [lane for lane in clear_lanes if lane != current_lane]
            escape_candidates = preferred_lanes if red_in_current_path and preferred_lanes else clear_lanes
            chosen_target_lane = min(
                escape_candidates,
                key=lambda lane: (abs(lane - current_lane), -abs(lane - red_lane), abs(lane))
            )
            decision_reason = "AVOID_RED_YELLOW"
            active_green_target = None
            active_yellow_target = None

    # Baseline Strategy
    if chosen_target_lane is None:
        chosen_target_lane = current_lane
        decision_reason = "MAINTAIN"

    if hazard_lanes:
        acceleration_input = min(acceleration_input, 0.66)

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
    # 3B. EVENT OVERRIDES (trailing car & police) — highest priority
    # =========================================================
    trailing_active = trailing_detected or (
        trailing_dodge_lane is not None and now < trailing_dodge_until
    )
    dodge_options = []

    # Trailing car: dodge to a locked edge lane until chaser clears
    if trailing_detected:
        # Liz added: Challenge 2 — dodge to a locked edge lane until chaser clears
        dodge_options = [l for l in [-2, -1, 0, 1, 2]
                         if l not in hazard_lanes and l != current_lane]
        if dodge_options:
            if (trailing_dodge_lane is None
                    or trailing_dodge_lane not in dodge_options
                    or trailing_dodge_lane == current_lane):
                edge_lanes = [l for l in dodge_options if abs(l) == 2]
                if edge_lanes:
                    trailing_dodge_lane = max(
                        edge_lanes,
                        key=lambda l: abs(l - current_lane)
                    )
                else:
                    trailing_dodge_lane = max(
                        dodge_options,
                        key=lambda l: abs(l - current_lane)
                    )
            chosen_target_lane = trailing_dodge_lane
            decision_reason = "TRAILING_CAR_DODGE"
            trailing_dodge_until = now + TRAILING_DODGE_HOLD_SECONDS
            notice = f"Dodge lane {chosen_target_lane}"
            if notice != last_trailing_notice:
                print(f"[CONTROL] Trailing car! Dodging to lane {chosen_target_lane}")
                last_trailing_notice = notice
        if debug_should_log('trailing'):
            debug_session_log('sample_drive.py:send_controls_task',
                          'TRAILING car detected -> dodge',
                          {'current_lane': current_lane,
                           'hazard_lanes': sorted(list(hazard_lanes)),
                           'chosen': chosen_target_lane,
                           'locked_dodge': trailing_dodge_lane,
                           'dodge_options': dodge_options},
                          'D',
                          run_id='verify')
    elif trailing_active and trailing_dodge_lane is not None:
        chosen_target_lane = trailing_dodge_lane
        decision_reason = "TRAILING_CAR_DODGE"
    elif not trailing_detected:
        trailing_dodge_lane = None
        trailing_dodge_until = 0.0
        last_trailing_notice = None

    # Liz added: Challenge 3 — seek nearest red token while police car is active
    active_red_target = None
    if police_detected:
        red_targets = [
            t for t in tokens_snapshot
            if t.get('color') == 'red' and t['y'] >= TOKEN_DECISION_Y_MIN
        ]
        if red_targets:
            active_red_target = max(red_targets, key=lambda t: t['y'])
            red_lane = lane_from_x(active_red_target['x'], active_red_target['y'],
                                   frame_width=img_w, frame_height=img_h)
            chosen_target_lane = max(-2, min(2, red_lane))
            decision_reason = "POLICE_SEEK_RED"
            acceleration_input = min(acceleration_input, POLICE_SEEK_ACCELERATION)
            notice = f"Seek red lane {chosen_target_lane}"
            if notice != last_police_notice:
                print(f"[CONTROL] Police detected! Seeking red token in lane {chosen_target_lane}")
                last_police_notice = notice
        else:
            acceleration_input = min(acceleration_input, POLICE_SEEK_ACCELERATION)
        if debug_should_log('police'):
            debug_session_log('sample_drive.py:send_controls_task',
                          'POLICE detected -> seek red token',
                          {'red_visible': len(red_targets),
                           'chosen': chosen_target_lane,
                           'current_lane': current_lane},
                          'E',
                          run_id='verify')
    else:
        last_police_notice = None

    event_override = decision_reason in ["TRAILING_CAR_DODGE", "POLICE_SEEK_RED"]
    if event_override and chosen_target_lane is not None:
        with data_lock:
            if shared_data['target_lane'] != chosen_target_lane:
                shared_data['target_lane'] = chosen_target_lane
                if steering_state == 1:
                    tap_loop_count = 0

    # =========================================================
    # 4. STEPPING CONTROL MOTOR ENGINE
    # =========================================================
    if decision_reason in [
        "COLLECT_GREEN", "GREEN_LOCK", "COLLECT_YELLOW", "YELLOW_LOCK",
        "AVOID_RED_YELLOW", "TRAILING_CAR_DODGE", "POLICE_SEEK_RED"
    ]:
        with data_lock:
            if shared_data['target_lane'] != chosen_target_lane:
                old_target_lane = shared_data['target_lane']
                shared_data['target_lane'] = chosen_target_lane
                old_direction = 0 if old_target_lane == current_lane else (1 if old_target_lane > current_lane else -1)
                new_direction = 0 if chosen_target_lane == current_lane else (1 if chosen_target_lane > current_lane else -1)
                if steering_state == 1 and old_direction != new_direction:
                    tap_loop_count = 0

    if steering_state == 0:
        if chosen_target_lane != current_lane:
            steering_state = 1
            tap_loop_count = 0
            with data_lock:
                shared_data['target_lane'] = chosen_target_lane

    elif steering_state == 1:
        with data_lock:
            tgt_lane = shared_data['target_lane']

        if tgt_lane == current_lane:
            steering_state = 0
            steering_input = 0.0
            steering_source = "lane_reached"
        else:
            steering_input = 1.0 if tgt_lane > current_lane else -1.0
            steering_source = "lane_change"
            
            lane_gap = abs(tgt_lane - current_lane)

            # Drop acceleration more for long cross-lane moves so the car has time to slide across.
            if decision_reason in ["COLLECT_GREEN", "GREEN_LOCK", "COLLECT_YELLOW", "YELLOW_LOCK"]:
                acceleration_input = min(acceleration_input, GREEN_CHASE_ACCELERATION - (0.08 if lane_gap >= 2 else 0.03))
            else:
                acceleration_input = min(acceleration_input, GREEN_CHASE_ACCELERATION - (0.08 if lane_gap >= 3 else 0.03))
            acceleration_input = max(MIN_STEERING_ACCELERATION, acceleration_input)
            
            tap_loop_count += 1
            phase_tap_loops = adaptive_tap_loops

            if tap_loop_count >= phase_tap_loops:
                steering_state = 2
                tap_loop_count = 0
                current_lane = max(-2, min(2, current_lane + (1 if tgt_lane > current_lane else -1)))

    elif steering_state == 2:
        steering_input = 0.0
        steering_source = "lane_reset"
        tap_loop_count += 1
        if tap_loop_count >= adaptive_reset_loops:
            tap_loop_count = 0
            with data_lock:
                tgt_lane = shared_data['target_lane']
            steering_state = 1 if tgt_lane != current_lane else 0

    if steering_state == 0 and decision_reason == "MAINTAIN":
        with data_lock:
            shared_data['target_lane'] = current_lane

    active_collect_target = active_green_target if active_green_target is not None else active_yellow_target
    if not event_override and active_collect_target is not None and active_collect_target['lane'] not in hazard_lanes:
        green_error = (active_collect_target['x'] - (img_w / 2.0)) / (img_w / 2.0)
        if abs(green_error) > TOKEN_CENTER_X_DEADZONE:
            visual_steer = max(-1.0, min(1.0, green_error * GREEN_STEER_GAIN))
            if abs(visual_steer) < GREEN_APPROACH_MIN_STEER:
                visual_steer = GREEN_APPROACH_MIN_STEER if visual_steer > 0 else -GREEN_APPROACH_MIN_STEER
            steering_input = visual_steer
            steering_source = "visual_green" if active_green_target is not None else "visual_yellow"
            acceleration_input = min(acceleration_input, GREEN_VISUAL_CORRECTION_ACCELERATION)
            if chosen_target_lane is not None and chosen_target_lane != current_lane:
                with data_lock:
                    shared_data['target_lane'] = chosen_target_lane
                if steering_state == 0:
                    steering_state = 1
                    tap_loop_count = 0
        elif chosen_target_lane is not None and chosen_target_lane != current_lane:
            steering_input = 1.0 if chosen_target_lane > current_lane else -1.0
            acceleration_input = max(
                GREEN_APPROACH_MIN_ACCELERATION,
                min(acceleration_input, GREEN_APPROACH_MAX_ACCELERATION)
            )
        else:
            if decision_reason in ["COLLECT_GREEN", "GREEN_LOCK", "COLLECT_YELLOW", "YELLOW_LOCK"]:
                steering_input = 0.0
                steering_source = "green_centered" if active_green_target is not None else "yellow_centered"
                acceleration_input = max(
                    GREEN_APPROACH_MIN_ACCELERATION,
                    min(acceleration_input, GREEN_CHASE_ACCELERATION)
                )
    elif decision_reason == "POLICE_SEEK_RED" and active_red_target is not None:
        red_error = (active_red_target['x'] - (img_w / 2.0)) / (img_w / 2.0)
        if abs(red_error) > TOKEN_CENTER_X_DEADZONE:
            visual_steer = max(-1.0, min(1.0, red_error * GREEN_STEER_GAIN))
            if abs(visual_steer) < GREEN_APPROACH_MIN_STEER:
                visual_steer = GREEN_APPROACH_MIN_STEER if visual_steer > 0 else -GREEN_APPROACH_MIN_STEER
            steering_input = visual_steer
            steering_source = "visual_red"
            acceleration_input = min(acceleration_input, POLICE_SEEK_ACCELERATION)
    elif decision_reason == "AVOID_RED_YELLOW" and closest_hazard is not None:
        hazard_error = (closest_hazard['x'] - (img_w / 2.0)) / (img_w / 2.0)
        if chosen_target_lane is not None and chosen_target_lane != current_lane:
            steering_input = HAZARD_AVOID_STEER if chosen_target_lane > current_lane else -HAZARD_AVOID_STEER
            steering_source = "red_escape_lane"
        elif abs(hazard_error) > TOKEN_CENTER_X_DEADZONE:
            steering_input = -HAZARD_AVOID_STEER if hazard_error > 0 else HAZARD_AVOID_STEER
            steering_source = "visual_hazard"
        elif closest_hazard['lane'] == current_lane:
            clear_lanes = [lane for lane in [-2, -1, 0, 1, 2] if lane not in hazard_lanes]
            if clear_lanes:
                escape_lane = min(clear_lanes, key=lambda lane: (abs(lane - current_lane), abs(lane)))
                steering_input = HAZARD_AVOID_STEER if escape_lane > current_lane else -HAZARD_AVOID_STEER
                steering_source = "hazard_lane"
        elif closest_hazard['lane'] > current_lane:
            steering_input = -HAZARD_AVOID_STEER
            steering_source = "hazard_lane"
        else:
            steering_input = HAZARD_AVOID_STEER
            steering_source = "hazard_lane"
        acceleration_input = min(acceleration_input, HAZARD_AVOID_MAX_ACCELERATION)

    with data_lock:
        hazard_debug = "none" if closest_hazard is None else f"{closest_hazard['color']} lane {closest_hazard['lane']} y {closest_hazard['y']}"
        green_x_debug = "none" if active_green_target is None else f"x {active_green_target['x']}"
        yellow_x_debug = "none" if active_yellow_target is None else f"x {active_yellow_target['x']}"
        center_x = img_w / 2.0
        green_error_debug = "none" if green_error is None else f"{green_error:.3f}"
        lock_debug = "ON" if locked_green_lane is not None and now < locked_green_until else "OFF"
        shared_data['decision_debug'] = f"Reason: {decision_reason} | Cur: {current_lane} | Target: {shared_data['target_lane']} | Green: {green_debug} {green_x_debug} | Yellow: {yellow_debug} {yellow_x_debug} | Cx {center_x:.0f} Err {green_error_debug} | Src: {steering_source} | Steer: {steering_input:.2f} | Accel: {acceleration_input:.2f} | Lock: {lock_debug}"

    if tokens_snapshot and now - last_token_debug_time >= 0.5:
        debug_parts = []
        for token in tokens_snapshot[:6]:
            token_lane = lane_from_x(token['x'], token['y'], frame_width=img_w, frame_height=img_h)
            debug_parts.append(f"{token.get('color')}@({token['x']},{token['y']})->L{token_lane}")
        print(
            "[DecisionDebug] "
            f"tokens={'; '.join(debug_parts)} | current={current_lane} | target={chosen_target_lane} | "
            f"green={green_debug} {green_x_debug} | yellow={yellow_debug} {yellow_x_debug} | "
            f"center_x={img_w / 2.0:.0f} | token_error={green_error_debug} | "
            f"visual_steer={visual_steer:.2f} | source={steering_source} | "
            f"steer={steering_input:.2f} | accel={acceleration_input:.2f} | reason={decision_reason} | lock={lock_debug}"
        )
        last_token_debug_time = now

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
    run_start_time = time.time()
    with data_lock:
        shared_data['target_lane'] = START_LANE
        # Liz added: reset challenge state at the start of each run
        shared_data['brightness_baseline'] = None
        shared_data['brightness_last'] = 0.0
        shared_data['low_light_active'] = False
        shared_data['low_light_recovered'] = False
        shared_data['trailing_streak'] = 0
        shared_data['police_streak'] = 0
        shared_data['lights_on'] = True

    cv2.namedWindow("Front Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Back Camera", cv2.WINDOW_NORMAL)
    with data_lock:
        shared_data['display_front_frame'] = make_camera_placeholder("Front Camera", "Starting controller...")
        shared_data['display_back_frame'] = make_camera_placeholder("Back Camera", "Starting controller...")
    cv2.moveWindow("Front Camera", 40, 40)
    cv2.moveWindow("Back Camera", 700, 40)
    update_camera_windows()

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
            update_camera_windows()
            time.sleep(0.01)
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