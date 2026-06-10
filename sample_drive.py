import socket
import threading
import struct
import cv2
import numpy as np
import time
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
TOKEN_DECISION_Y_MIN = 70
TOKEN_TARGET_MEMORY_SECONDS = 0.65
GREEN_TARGET_LOCK_SECONDS = 0.65
YELLOW_EFFECT_DURATION_SECONDS = 5.0
TOKEN_COLLECTION_Y_RATIO = 0.78
TOKEN_COLLECTION_COOLDOWN_SECONDS = 0.7
CAMERA_DELAY_SECONDS = 5.0
ACTION_DELAY_SECONDS = 5.0

# Five-lane token targeting
# Lane changes must use tap-based steering only: neutral -> +/-1.0 tap -> neutral.
LANE_CHANGE_PRE_TAP_LOOPS = 2
LANE_CHANGE_TAP_LOOPS = 24
LANE_CHANGE_RESET_LOOPS = 4
LANE_CHANGE_TIME_Y_RATIO = 0.14
GREEN_REACHABILITY_BASE_MARGIN = 0.035
CAR_PICKUP_X_RATIO = 0.50

# Hazard avoidance during normal driving. Red and yellow tokens remain detected,
# but the token-only controller does not intentionally collect them.
UNWANTED_TOKEN_AVOID_Y_RATIO = 0.52
YELLOW_HAZARD_PENALTY = 10.0
RED_HAZARD_PENALTY = 8.0
LANE_SWITCH_PENALTY = 0.65
GREEN_LANE_REWARD = 3.0

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
    'steering_input': 0.0,
    'acceleration_input': 0.0,
    'detected_tokens': [],
    'target_lane': START_LANE,
    'decision_debug': '',
    'target_token_debug': None,
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
        }
    },
    'active_yellow_effect': None,
    'yellow_effect_until': 0.0,
    'hidden_next_token_type': False,
    'last_collected_time': 0.0,
    'last_collected_line_id': None,
    'last_collected_by_lane_color': {}
}
data_lock = threading.Lock()
is_running = True
front_camera_delay_buffer = []
action_delay_buffer = []

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
    """
    Connect both camera streams.

    The front camera is used for token processing. The back camera is retained
    as a live preview only so a teammate can add rear-event handling later.
    """
    global front_camera_sock, back_camera_sock

    print("Connecting to Cameras...")
    front_connected = False
    back_connected = False

    while is_running and not (front_connected and back_connected):
        if not front_connected:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                sock.connect((CAMERA_HOST, FRONT_CAMERA_PORT))
                front_camera_sock = sock
                front_connected = True
                print("Connected to Front Camera successfully.")
            except Exception:
                pass

        if not back_connected:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                sock.connect((CAMERA_HOST, BACK_CAMERA_PORT))
                back_camera_sock = sock
                back_connected = True
                print("Connected to Back Camera successfully.")
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
    # Preview only: rear-event detection will be added separately by the team.
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


def calculate_token_distance(token, frame_width, frame_height):
    """
    Estimate the token's distance from the car in normalised image space.

    This is not a distance in metres. A calibrated homography would be needed
    for metric ground-plane distance. For target ranking and steering control,
    normalised image-space distance is sufficient and more robust than raw
    pixel distance across different resolutions.
    """
    pickup_x = frame_width * CAR_PICKUP_X_RATIO
    pickup_y = frame_height * TOKEN_COLLECTION_Y_RATIO

    lateral_error_ratio = (token['x'] - pickup_x) / max(frame_width / 2.0, 1.0)
    remaining_y_ratio = (pickup_y - token['y']) / max(float(frame_height), 1.0)
    remaining_y_ratio = max(0.0, remaining_y_ratio)

    image_distance = float(np.hypot(lateral_error_ratio, remaining_y_ratio))
    return {
        'image_distance': image_distance,
        'lateral_error_ratio': float(lateral_error_ratio),
        'remaining_y_ratio': float(remaining_y_ratio),
        'pickup_x': float(pickup_x),
        'pickup_y': float(pickup_y)
    }


def evaluate_token_candidate(token, current_vehicle_lane, frame_width, frame_height):
    """
    Add lane-gap, reachability, and distance information to one detected token.

    A lane change is considered feasible only when enough vertical image-space
    travel remains before the token reaches the collection line. This prevents
    the controller from starting an impossible late lane change.
    """
    lane = lane_from_x(token['x'], token['y'], frame_width=frame_width, frame_height=frame_height)
    lane = max(-2, min(2, lane))
    lane_gap = abs(lane - current_vehicle_lane)
    metrics = calculate_token_distance(token, frame_width, frame_height)

    required_margin = GREEN_REACHABILITY_BASE_MARGIN + lane_gap * LANE_CHANGE_TIME_Y_RATIO
    reachable = lane_gap == 0 or metrics['remaining_y_ratio'] >= required_margin

    # Prefer the next reachable token, while slightly penalising risky long
    # cross-lane movements and large lateral errors.
    progress_ratio = token['y'] / max(float(frame_height), 1.0)
    route_score = (
        progress_ratio * 3.0
        - lane_gap * 0.16
        - abs(metrics['lateral_error_ratio']) * 0.08
    )

    return {
        'token': dict(token),
        'lane': lane,
        'lane_gap': lane_gap,
        'color': token.get('color'),
        'y': token['y'],
        'reachable': reachable,
        'route_score': float(route_score),
        **metrics
    }



def choose_low_risk_lane(tokens, current_vehicle_lane, frame_width, frame_height,
                         candidate_lanes=None):
    """
    Choose the safest lane when no reachable green target exists.

    Red and yellow tokens remain visible to perception, but this token-only
    controller uses them as hazards rather than intentional collection targets.
    """
    if candidate_lanes is None:
        candidate_lanes = list(range(-2, 3))

    best_lane = current_vehicle_lane
    best_score = -float('inf')

    for lane in candidate_lanes:
        lane = max(-2, min(2, lane))
        score = -abs(lane - current_vehicle_lane) * LANE_SWITCH_PENALTY

        for token in tokens:
            color = token.get('true_color', token.get('color'))
            if color not in ['green', 'yellow', 'red']:
                continue

            token_lane = lane_from_x(
                token['x'], token['y'],
                frame_width=frame_width,
                frame_height=frame_height
            )
            if token_lane != lane:
                continue

            progress = token['y'] / max(float(frame_height), 1.0)
            proximity_weight = 1.0 + 4.0 * progress * progress

            if color == 'green':
                score += GREEN_LANE_REWARD * proximity_weight
            elif progress >= UNWANTED_TOKEN_AVOID_Y_RATIO:
                if color == 'yellow':
                    score -= YELLOW_HAZARD_PENALTY * proximity_weight
                elif color == 'red':
                    score -= RED_HAZARD_PENALTY * proximity_weight

        if score > best_score:
            best_score = score
            best_lane = lane

    return best_lane


def select_best_green_candidate(candidates, tokens, current_vehicle_lane,
                                frame_width, frame_height):
    """
    Select a reachable green target using a lane-route score.

    The immediate token remains important, but the score also rewards other
    visible greens in the same lane and penalises nearby red/yellow hazards.
    This is more useful than selecting only the closest contour.
    """
    if not candidates:
        return None

    lane_scores = {}
    for lane in range(-2, 3):
        score = -abs(lane - current_vehicle_lane) * LANE_SWITCH_PENALTY
        for token in tokens:
            color = token.get('true_color', token.get('color'))
            if color not in ['green', 'yellow', 'red']:
                continue

            token_lane = lane_from_x(
                token['x'], token['y'],
                frame_width=frame_width,
                frame_height=frame_height
            )
            if token_lane != lane:
                continue

            progress = token['y'] / max(float(frame_height), 1.0)
            proximity_weight = 1.0 + 4.0 * progress * progress
            if color == 'green':
                score += GREEN_LANE_REWARD * proximity_weight
            elif progress >= UNWANTED_TOKEN_AVOID_Y_RATIO:
                if color == 'yellow':
                    score -= YELLOW_HAZARD_PENALTY * proximity_weight
                elif color == 'red':
                    score -= RED_HAZARD_PENALTY * proximity_weight
        lane_scores[lane] = score

    return max(
        candidates,
        key=lambda candidate: (
            lane_scores.get(candidate['lane'], -float('inf')),
            candidate['route_score']
        )
    )


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
    """Estimate accidental or intentional token collection for local effects."""
    global locked_green_lane, locked_green_until

    now = time.time()
    collection_y = frame_height * TOKEN_COLLECTION_Y_RATIO
    collected = None

    for token in tokens:
        if token['y'] < collection_y:
            continue

        token_lane = lane_from_x(
            token['x'], token['y'],
            frame_width=frame_width,
            frame_height=frame_height
        )
        if token_lane != current_lane:
            continue

        color = token.get('true_color', token.get('color'))
        if color not in ['green', 'yellow', 'red']:
            continue

        token_id = f"{token_lane}:{color}"
        with data_lock:
            recent_collections = shared_data.get('last_collected_by_lane_color', {})
            recent_same_token = (
                now - recent_collections.get(token_id, 0.0)
                < TOKEN_COLLECTION_COOLDOWN_SECONDS
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
        # Yellow is never an intentional target, but an accidental pickup still
        # activates the required random disruption model.
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
    """
    Detect tokens from the front camera and display both camera previews.

    Rear-camera frames are intentionally not analysed in this token-only
    version. They are displayed so rear-event processing can be integrated
    later without changing the camera connection architecture again.
    """
    with data_lock:
        front_frame = shared_data.get('latest_front_frame')
        back_frame = shared_data.get('latest_back_frame')

    # Show an unmodified rear preview. Yellow camera effects apply to the
    # front perception input only because token detection uses the front view.
    if back_frame is not None:
        back_debug_frame = cv2.resize(back_frame, (640, 480))
        cv2.putText(
            back_debug_frame, "Back Camera Preview Only", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2
        )
        cv2.imshow("Back Camera", back_debug_frame)
        cv2.waitKey(1)

    if front_frame is None:
        return

    processed_front_frame = apply_camera_effects(front_frame)
    tokens = detect_colored_tokens(processed_front_frame)
    tokens = apply_token_visibility_effects(tokens)

    with data_lock:
        shared_data['detected_tokens'] = tokens

    debug_frame = draw_detected_tokens(processed_front_frame, tokens)
    with data_lock:
        decision_text = shared_data.get('decision_debug', '')
        target_debug = shared_data.get('target_token_debug')

    if decision_text:
        cv2.putText(
            debug_frame, decision_text, (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2
        )

    if target_debug is not None:
        target_token = target_debug.get('token', {})
        tx = int(target_token.get('x', 0))
        ty = int(target_token.get('y', 0))
        cv2.circle(debug_frame, (tx, ty), 10, (255, 255, 255), 2)
        cv2.putText(
            debug_frame,
            f"TARGET D={target_debug.get('image_distance', 0.0):.3f}",
            (max(tx - 85, 0), max(ty - 24, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2
        )

    effect = get_active_yellow_effect()
    if effect:
        cv2.putText(
            debug_frame, f"Yellow Effect: {effect}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

    debug_frame = cv2.resize(debug_frame, (640, 480))
    cv2.imshow("Front Camera", debug_frame)
    cv2.waitKey(1)


# Global variables for strict tap-based lane changes
# Sequence for each one-lane movement:
#   PRE_TAP_NEUTRAL (0.0) -> TAP_ACTIVE (+/-1.0) -> RESET_NEUTRAL (0.0)
TAP_IDLE = 0
TAP_PRE_TAP_NEUTRAL = 1
TAP_ACTIVE = 2
TAP_RESET_NEUTRAL = 3

steering_state = TAP_IDLE
tap_loop_count = 0
tap_direction = 0.0
current_lane = START_LANE
last_token_lane = None
last_token_time = 0.0
locked_green_lane = None
locked_green_until = 0.0
run_start_time = None


def send_controls_task():
    """
    Choose a token lane and send tap-based steering commands only.

    Steering output is deliberately restricted to -1.0, 0.0, or +1.0.
    No proportional steering or continuous fine-alignment command is used.
    """
    global control_conn
    global steering_state, tap_loop_count, tap_direction, current_lane
    global last_token_lane, last_token_time, locked_green_lane, locked_green_until
    global run_start_time

    if control_conn is None:
        return

    with data_lock:
        tokens_snapshot = list(shared_data.get('detected_tokens', []))
        front_frame = shared_data.get('latest_front_frame')
        run_summary = dict(shared_data.get('run_summary', {}))

    steering_input = 0.0
    acceleration_input = CAR_ACCELERATION

    if run_start_time is not None and time.time() - run_start_time < START_CENTER_HOLD_SECONDS:
        try:
            send_control_packet(0.0, 0.0)
        except Exception:
            pass
        current_lane = START_LANE
        return

    img_w, img_h = (
        (front_frame.shape[1], front_frame.shape[0])
        if front_frame is not None else (640, 480)
    )
    maybe_record_collected_token(tokens_snapshot, img_w, img_h)

    green_collected = run_summary.get('green_collected', 0)
    red_collected = run_summary.get('red_collected', 0)

    # Local estimate only. The simulator remains the source of truth.
    estimated_speed_modifier = 1.0 + green_collected * 0.10 - red_collected * 0.20
    estimated_speed_modifier = max(0.5, min(2.5, estimated_speed_modifier))
    acceleration_input = max(0.25, min(1.0, CAR_ACCELERATION * estimated_speed_modifier))

    # Preserve tap semantics while allowing a modest duration adjustment as the
    # simulated speed changes.
    adaptive_pre_tap_loops = max(
        1, int(LANE_CHANGE_PRE_TAP_LOOPS / (estimated_speed_modifier ** 0.25))
    )
    adaptive_tap_loops = max(
        8, int(LANE_CHANGE_TAP_LOOPS / (estimated_speed_modifier ** 0.25))
    )
    adaptive_reset_loops = max(
        2, int(LANE_CHANGE_RESET_LOOPS / (estimated_speed_modifier ** 0.25))
    )

    # Detect all colours, but only green tokens become collection candidates.
    evaluated_tokens = []
    green_candidates = []
    for token in tokens_snapshot:
        color = token.get('true_color', token.get('color'))
        if color not in ['green', 'yellow', 'red']:
            continue

        candidate = evaluate_token_candidate(
            token, current_lane, img_w, img_h
        )
        candidate['color'] = color
        evaluated_tokens.append(candidate)

        if (
            color == 'green'
            and token['y'] >= TOKEN_DECISION_Y_MIN
            and candidate['reachable']
        ):
            green_candidates.append(candidate)

    chosen_target_lane = None
    active_green_candidate = None
    decision_reason = 'MAINTAIN_SAFE_LANE'
    now = time.time()

    # Retain a short lock to prevent rapid lane oscillation, but allow a new
    # visible green to replace a stale lock immediately.
    if locked_green_lane is not None and now < locked_green_until:
        same_lane_greens = [
            candidate for candidate in green_candidates
            if candidate['lane'] == locked_green_lane
        ]
        active_green_candidate = select_best_green_candidate(
            same_lane_greens, tokens_snapshot, current_lane, img_w, img_h
        )

    if active_green_candidate is None:
        active_green_candidate = select_best_green_candidate(
            green_candidates, tokens_snapshot, current_lane, img_w, img_h
        )

    if active_green_candidate is not None:
        chosen_target_lane = active_green_candidate['lane']
        locked_green_lane = chosen_target_lane
        locked_green_until = now + GREEN_TARGET_LOCK_SECONDS
        last_token_lane = chosen_target_lane
        last_token_time = now
        decision_reason = 'COLLECT_GREEN'
    elif last_token_lane is not None and now - last_token_time < TOKEN_TARGET_MEMORY_SECONDS:
        # Brief memory prevents a lane change from being aborted by one missed
        # contour, while the short timeout limits stale-target behaviour.
        chosen_target_lane = last_token_lane
        decision_reason = 'GREEN_MEMORY'
    else:
        chosen_target_lane = choose_low_risk_lane(
            tokens_snapshot, current_lane, img_w, img_h
        )
        decision_reason = (
            'AVOID_RED_YELLOW'
            if chosen_target_lane != current_lane
            else 'MAINTAIN_SAFE_LANE'
        )

    chosen_target_lane = max(-2, min(2, chosen_target_lane))
    with data_lock:
        shared_data['target_lane'] = chosen_target_lane

    # Strict tap state machine. Each one-lane move always includes a neutral
    # period before and after the +/-1.0 steering pulse.
    if steering_state == TAP_IDLE:
        steering_input = 0.0
        if chosen_target_lane != current_lane:
            steering_state = TAP_PRE_TAP_NEUTRAL
            tap_loop_count = 0

    elif steering_state == TAP_PRE_TAP_NEUTRAL:
        steering_input = 0.0
        tap_loop_count += 1
        if tap_loop_count >= adaptive_pre_tap_loops:
            tap_loop_count = 0
            if chosen_target_lane == current_lane:
                steering_state = TAP_IDLE
            else:
                tap_direction = 1.0 if chosen_target_lane > current_lane else -1.0
                steering_state = TAP_ACTIVE

    elif steering_state == TAP_ACTIVE:
        steering_input = tap_direction
        acceleration_input = min(acceleration_input, GREEN_CHASE_ACCELERATION)
        tap_loop_count += 1
        if tap_loop_count >= adaptive_tap_loops:
            tap_loop_count = 0
            current_lane = max(-2, min(2, current_lane + int(tap_direction)))
            steering_state = TAP_RESET_NEUTRAL

    elif steering_state == TAP_RESET_NEUTRAL:
        steering_input = 0.0
        tap_loop_count += 1
        if tap_loop_count >= adaptive_reset_loops:
            tap_loop_count = 0
            tap_direction = 0.0
            steering_state = TAP_IDLE

    target_debug = None
    debug_suffix = ''
    if active_green_candidate is not None:
        target_debug = dict(active_green_candidate)
        debug_suffix = (
            f" | Dist: {active_green_candidate['image_distance']:.3f}"
            f" | RemY: {active_green_candidate['remaining_y_ratio']:.3f}"
        )

    state_names = {
        TAP_IDLE: 'IDLE',
        TAP_PRE_TAP_NEUTRAL: 'PRE_NEUTRAL',
        TAP_ACTIVE: 'TAP',
        TAP_RESET_NEUTRAL: 'RESET_NEUTRAL'
    }
    with data_lock:
        shared_data['steering_input'] = steering_input
        shared_data['acceleration_input'] = acceleration_input
        shared_data['decision_debug'] = (
            f"Reason: {decision_reason} | GoTo: {chosen_target_lane}"
            f" | Lane: {current_lane} | TapState: {state_names[steering_state]}"
            f"{debug_suffix}"
        )
        shared_data['target_token_debug'] = target_debug

    try:
        send_control_packet(steering_input, acceleration_input)
    except Exception as exc:
        print(f"Network error: {exc}")
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

    # Initialize front camera, back-camera preview, and control connections
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
    print("="*45 + "\n")
