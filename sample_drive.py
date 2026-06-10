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
CAR_ACCELERATION = 0.78
GREEN_CHASE_ACCELERATION = 0.62
GREEN_STEER_GAIN = 2.8
GREEN_STEER_DEADZONE = 0.015
GREEN_MIN_STEER = 0.18
MIN_LOOKAHEAD_Y_RATIO = 0.25
LINE_GROUP_Y_RATIO = 0.08
LANE_CHANGE_TIME_Y_RATIO = 0.14
GREEN_LOCK_SECONDS = 1.6
STEER_TAP_LOOPS = 12
STEER_RESET_LOOPS = 4
TOKEN_DECISION_Y_MIN = 70
LANE_CHANGE_TAP_LOOPS = 24
LANE_CHANGE_RESET_LOOPS = 2
TOKEN_TARGET_MEMORY_SECONDS = 1.8
GREEN_TARGET_LOCK_SECONDS = 2.2
YELLOW_EFFECT_DURATION_SECONDS = 5.0
TOKEN_COLLECTION_Y_RATIO = 0.78
TOKEN_COLLECTION_COOLDOWN_SECONDS = 0.7
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
    'decision_debug': ''
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
            min_circularity = 0.50 if color_name == "green" else 0.75
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
        color = text_colors.get(token["color"], (255, 255, 255))
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
        cv2.putText(display_frame, token["color"].upper(), (x1, y1-10),
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
        shared_data['active_yellow_effect'] = effect
        shared_data['yellow_effect_until'] = now + YELLOW_EFFECT_DURATION_SECONDS
        shared_data['run_summary']['yellow_effects'][effect] += 1
        if effect == 'hide_next_token_type':
            shared_data['hidden_next_token_type'] = True

def maybe_record_collected_token(tokens, frame_width, frame_height):
    global locked_green_lane, locked_green_until

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
    hidden_tokens[closest_index]['true_color'] = hidden_tokens[closest_index]['color']
    hidden_tokens[closest_index]['color'] = 'hidden'

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
    with data_lock:
        front_frame = shared_data['latest_front_frame']
    
    with data_lock:
        back_frame = shared_data.get('latest_back_frame')

    if front_frame is not None:
        processed_front_frame = apply_camera_effects(front_frame)
        tokens = detect_colored_tokens(processed_front_frame)
        tokens = apply_token_visibility_effects(tokens)

        with data_lock:
            shared_data['detected_tokens'] = tokens

        debug_frame = draw_detected_tokens(processed_front_frame, tokens)
        effect = get_active_yellow_effect()
        if effect:
            cv2.putText(debug_frame, f"Yellow Effect: {effect}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
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
run_start_time = None
def send_controls_task():
    global control_conn, steering_state, tap_loop_count, current_lane, last_token_lane, last_token_time, locked_green_lane, locked_green_until, run_start_time
    
    if control_conn is None:
        return
    
    with data_lock:
        tokens_snapshot = list(shared_data.get('detected_tokens', []))
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

    # =========================================================
    # 1. SPEED MODIFIER & TUNING PARAMETERS
    # =========================================================
    estimated_speed_modifier = 1.0 + (green_streak * 0.10) - (red_streak * 0.20)
    estimated_speed_modifier = max(0.5, min(2.5, estimated_speed_modifier))
    acceleration_input = max(0.25, min(1.0, CAR_ACCELERATION * estimated_speed_modifier))

    # Calculate adaptive state machine loops based on velocity.
    adaptive_tap_loops = max(10, int(LANE_CHANGE_TAP_LOOPS / (estimated_speed_modifier ** 0.25)))
    adaptive_reset_loops = max(1, int(LANE_CHANGE_RESET_LOOPS / (estimated_speed_modifier ** 0.25)))

    # =========================================================
    # 2. TOKEN PRIORITY CLASSIFICATION
    # =========================================================
    token_priority = {'yellow': 2, 'red': 1}
    best_green = None
    best_backup_token = None

    for t in tokens_snapshot:
        t_lane = lane_from_x(t['x'], t['y'], frame_width=img_w, frame_height=img_h)
        t_lane = max(-2, min(2, t_lane))
        color = t.get('color')

        if color not in ['green', 'yellow', 'red']:
            continue

        # Decide early, when the token is still far enough away to reach.
        if t['y'] < TOKEN_DECISION_Y_MIN:
            continue

        candidate = {
            'lane': t_lane,
            'y': t['y'],
            'color': color,
            'score': 3 if color == 'green' else token_priority[color]
        }

        if color == 'green':
            if best_green is None or candidate['y'] < best_green['y']:
                best_green = candidate
        elif best_backup_token is None:
            best_backup_token = candidate
        elif candidate['score'] > best_backup_token['score']:
            best_backup_token = candidate
        elif candidate['score'] == best_backup_token['score'] and candidate['y'] < best_backup_token['y']:
            best_backup_token = candidate

    # =========================================================
    # 3. GREEN-FIRST TARGET DECISION
    # =========================================================
    chosen_target_lane = None
    decision_reason = "MAINTAIN"

    now = time.time()

    if best_green is not None:
        chosen_target_lane = best_green['lane']
        locked_green_lane = chosen_target_lane
        locked_green_until = now + GREEN_TARGET_LOCK_SECONDS
        decision_reason = "COLLECT_GREEN"
        last_token_lane = chosen_target_lane
        last_token_time = now
    elif locked_green_lane is not None and now < locked_green_until:
        chosen_target_lane = locked_green_lane
        decision_reason = "GREEN_LOCK"
    elif best_backup_token is not None:
        chosen_target_lane = best_backup_token['lane']
        decision_reason = f"COLLECT_{best_backup_token['color'].upper()}"
        last_token_lane = chosen_target_lane
        last_token_time = now

    # Memory Fallback: Prevent aborting mid-maneuver due to camera sensor noise
    if chosen_target_lane is None and last_token_lane is not None:
        if now - last_token_time < TOKEN_TARGET_MEMORY_SECONDS:
            chosen_target_lane = last_token_lane
            decision_reason = "TOKEN_MEMORY"

    # Baseline Strategy
    if chosen_target_lane is None:
        chosen_target_lane = current_lane

    # =========================================================
    # 4. STEPPING CONTROL MOTOR ENGINE
    # =========================================================
    if decision_reason in ["COLLECT_GREEN", "GREEN_LOCK"]:
        with data_lock:
            if shared_data['target_lane'] != chosen_target_lane:
                shared_data['target_lane'] = chosen_target_lane

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
        else:
            steering_input = 1.0 if tgt_lane > current_lane else -1.0
            
            lane_gap = abs(tgt_lane - current_lane)

            # Drop acceleration more for long cross-lane moves so the car has time to slide across.
            acceleration_input = min(acceleration_input, GREEN_CHASE_ACCELERATION - (0.12 if lane_gap >= 3 else 0.06))
            
            tap_loop_count += 1
            if tap_loop_count >= adaptive_tap_loops:
                steering_state = 2
                tap_loop_count = 0
                current_lane = max(-2, min(2, current_lane + (1 if tgt_lane > current_lane else -1)))

    elif steering_state == 2:
        steering_input = 0.0
        tap_loop_count += 1
        if tap_loop_count >= adaptive_reset_loops:
            tap_loop_count = 0
            with data_lock:
                tgt_lane = shared_data['target_lane']
            steering_state = 1 if tgt_lane != current_lane else 0

    with data_lock:
        shared_data['decision_debug'] = f"Reason: {decision_reason} | GoTo: {chosen_target_lane}"

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