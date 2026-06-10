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
TOKEN_TARGET_MEMORY_SECONDS = 0.22
GREEN_TARGET_LOCK_SECONDS = 0.35
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
# but the token-only controller does not intentionally collect them. Lane values
# from the camera are RELATIVE offsets and must be converted before planning.
UNWANTED_TOKEN_AVOID_Y_RATIO = 0.30
EMERGENCY_HAZARD_Y_RATIO = 0.42
EMERGENCY_ESCAPE_ACCELERATION = 0.34
EMERGENCY_BRAKE_ACCELERATION = 0.24
YELLOW_EFFECT_SAFE_ACCELERATION = 0.38
YELLOW_HAZARD_PENALTY = 20.0
RED_HAZARD_PENALTY = 18.0
LANE_SWITCH_PENALTY = 0.35
GREEN_LANE_REWARD = 2.5

# Accuracy improvements -------------------------------------------------------
# Green tokens must persist across new camera frames before the car chases them.
# Two frames is a good initial balance between recall and false-positive control.
GREEN_CONFIRM_FRAMES = 2
TOKEN_TRACK_MAX_MISSES = 4
TOKEN_TRACK_MAX_AGE_SECONDS = 0.45
TOKEN_TRACK_MATCH_DISTANCE_RATIO = 0.075
STALE_FRONT_FRAME_SECONDS = 0.35

# A hidden type must be treated as dangerous. Do not use its internal true colour
# when selecting a lane; the true colour is retained only for local bookkeeping.
UNKNOWN_HAZARD_PENALTY = 24.0
HAZARD_BLOCK_Y_MARGIN_RATIO = 0.10

# Road-only region of interest. Tune these ratios from a simulator screenshot if
# the visible road geometry changes. The polygon is intentionally broad initially.
ROAD_ROI_TOP_Y_RATIO = 0.10
ROAD_ROI_TOP_LEFT_X_RATIO = 0.30
ROAD_ROI_TOP_RIGHT_X_RATIO = 0.70
ROAD_ROI_BOTTOM_Y_RATIO = 0.99
ROAD_ROI_BOTTOM_LEFT_X_RATIO = 0.00
ROAD_ROI_BOTTOM_RIGHT_X_RATIO = 1.00
SHOW_ROAD_ROI = True
SHOW_TOKEN_MASKS = False

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
    'front_frame_seq': 0,
    'front_frame_time': 0.0,
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

# Token tracks are updated only by the processing task when a new camera frame
# arrives. The control task reads the resulting snapshot from shared_data.
token_tracks = {}
next_token_track_id = 1
last_processed_front_seq = -1

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
                    if data_key == 'latest_front_frame':
                        shared_data['front_frame_seq'] += 1
                        shared_data['front_frame_time'] = time.monotonic()
                
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
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)


def get_visible_color(token):
    """Return only the colour that perception is currently allowed to use."""
    return token.get('color')


def get_actual_color(token):
    """Return the actual colour for collection statistics and yellow effects only."""
    return token.get('true_color', token.get('color'))


def create_road_roi(frame_h, frame_w):
    """Create a broad trapezoidal mask covering the drivable road surface."""
    points = np.array([
        [int(frame_w * ROAD_ROI_TOP_LEFT_X_RATIO),
         int(frame_h * ROAD_ROI_TOP_Y_RATIO)],
        [int(frame_w * ROAD_ROI_TOP_RIGHT_X_RATIO),
         int(frame_h * ROAD_ROI_TOP_Y_RATIO)],
        [int(frame_w * ROAD_ROI_BOTTOM_RIGHT_X_RATIO),
         int(frame_h * ROAD_ROI_BOTTOM_Y_RATIO)],
        [int(frame_w * ROAD_ROI_BOTTOM_LEFT_X_RATIO),
         int(frame_h * ROAD_ROI_BOTTOM_Y_RATIO)]
    ], dtype=np.int32)

    roi = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillPoly(roi, [points], 255)
    return roi


def passes_round_token_shape(cnt, frame_w, frame_h, color_name):
    """Validate a candidate using scaled area, roundness and circle fill ratio."""
    area = cv2.contourArea(cnt)
    frame_area = float(frame_w * frame_h)

    if color_name == 'red':
        min_area = max(55.0, frame_area * 0.00018)
        min_circularity = 0.46
        min_fill_ratio = 0.43
        min_aspect, max_aspect = 0.58, 1.55
    elif color_name == 'yellow':
        min_area = max(35.0, frame_area * 0.00010)
        min_circularity = 0.62
        min_fill_ratio = 0.47
        min_aspect, max_aspect = 0.62, 1.48
    else:  # green
        min_area = max(35.0, frame_area * 0.00010)
        min_circularity = 0.48
        min_fill_ratio = 0.40
        min_aspect, max_aspect = 0.58, 1.55

    max_area = frame_area * 0.18
    if area < min_area or area > max_area:
        return None

    x, y, width, height = cv2.boundingRect(cnt)
    if height <= 0:
        return None

    aspect_ratio = width / float(height)
    if not (min_aspect <= aspect_ratio <= max_aspect):
        return None

    perimeter = cv2.arcLength(cnt, True)
    if perimeter <= 0:
        return None

    circularity = 4.0 * np.pi * area / (perimeter * perimeter)
    if circularity < min_circularity:
        return None

    (center_x, center_y), radius = cv2.minEnclosingCircle(cnt)
    if radius < 3.0:
        return None

    fill_ratio = area / (np.pi * radius * radius)
    if fill_ratio < min_fill_ratio:
        return None

    # Exclude HUD fragments and extreme lower-edge fragments.
    if center_y < frame_h * 0.08 or center_y > frame_h * 0.94:
        return None

    return {
        'color': color_name,
        'x': int(center_x),
        'y': int(center_y),
        'radius': int(radius),
        'area': float(area),
        'circularity': float(circularity),
        'fill_ratio': float(fill_ratio)
    }


def detect_colored_tokens(frame):
    """
    Detect green, yellow and red circular tokens inside the drivable road ROI.

    Green/yellow masks are generated from a gamma-corrected image. Red is
    generated from the original image because pale red gradients are easier to
    retain before brightness correction.
    """
    frame_h, frame_w = frame.shape[:2]
    road_roi = create_road_roi(frame_h, frame_w)

    frame_bright = adjust_gamma(frame, gamma=1.5)
    hsv_bright = cv2.cvtColor(frame_bright, cv2.COLOR_BGR2HSV)
    hsv_original = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    detected_tokens = []
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Keep the HSV values easy to tune after testing recorded simulator frames.
    standard_color_ranges = {
        'green': [((50, 30, 150), (70, 255, 255))],
        'yellow': [((18, 50, 140), (30, 255, 255))]
    }

    debug_masks = {}

    for color_name, ranges in standard_color_ranges.items():
        color_mask = None
        for lower, upper in ranges:
            current_mask = cv2.inRange(
                hsv_bright,
                np.array(lower, dtype=np.uint8),
                np.array(upper, dtype=np.uint8)
            )
            color_mask = (
                current_mask if color_mask is None
                else cv2.bitwise_or(color_mask, current_mask)
            )

        # Closing reconnects fragmented token pixels; the smaller opening
        # kernel suppresses noise without deleting small distant tokens.
        color_mask = cv2.bitwise_and(color_mask, road_roi)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, close_kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, open_kernel)
        debug_masks[color_name] = color_mask

        contours, _ = cv2.findContours(
            color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            token = passes_round_token_shape(cnt, frame_w, frame_h, color_name)
            if token is not None:
                detected_tokens.append(token)

    # OpenCV hue wraps around: red is near both 0 and 179.
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
    red_mask = cv2.bitwise_and(red_mask, road_roi)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, close_kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, open_kernel)
    debug_masks['red'] = red_mask

    red_contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in red_contours:
        token = passes_round_token_shape(cnt, frame_w, frame_h, 'red')
        if token is not None:
            detected_tokens.append(token)

    if SHOW_TOKEN_MASKS:
        for color_name, mask in debug_masks.items():
            cv2.imshow(
                f'{color_name.title()} Token Mask',
                cv2.resize(mask, (640, 480))
            )

    return detected_tokens


def update_token_tracks(detections, frame_width, frame_height):
    """
    Add stable track IDs and consecutive-frame hit counts.

    The controller chases a green token only after GREEN_CONFIRM_FRAMES hits.
    Hazard colours remain immediately usable so red/yellow avoidance is fast.
    """
    global next_token_track_id

    now = time.monotonic()
    unmatched_track_ids = set(token_tracks.keys())
    tracked_detections = []

    # Nearest objects are matched first because they are the most safety-critical.
    for detection in sorted(detections, key=lambda item: item['y'], reverse=True):
        actual_color = get_actual_color(detection)
        detection_lane = lane_from_x(
            detection['x'], detection['y'],
            frame_width=frame_width,
            frame_height=frame_height
        )

        best_track_id = None
        best_distance = float('inf')

        for track_id in unmatched_track_ids:
            track = token_tracks[track_id]
            if track['actual_color'] != actual_color:
                continue

            track_lane = lane_from_x(
                track['x'], track['y'],
                frame_width=frame_width,
                frame_height=frame_height
            )
            if abs(track_lane - detection_lane) > 1:
                continue

            distance = float(np.hypot(
                detection['x'] - track['x'],
                detection['y'] - track['y']
            ))
            match_limit = max(
                24.0,
                frame_width * TOKEN_TRACK_MATCH_DISTANCE_RATIO,
                3.0 * max(detection['radius'], track['radius'])
            )

            if distance <= match_limit and distance < best_distance:
                best_track_id = track_id
                best_distance = distance

        if best_track_id is None:
            track_id = next_token_track_id
            next_token_track_id += 1
            previous_y = detection['y']
            token_tracks[track_id] = {
                'x': detection['x'],
                'y': detection['y'],
                'radius': detection['radius'],
                'actual_color': actual_color,
                'hits': 1,
                'misses': 0,
                'last_seen': now
            }
        else:
            track_id = best_track_id
            track = token_tracks[track_id]
            previous_y = track['y']
            track.update({
                'x': detection['x'],
                'y': detection['y'],
                'radius': detection['radius'],
                'actual_color': actual_color,
                'hits': track['hits'] + 1,
                'misses': 0,
                'last_seen': now
            })
            unmatched_track_ids.remove(track_id)

        enriched = dict(detection)
        enriched['track_id'] = track_id
        enriched['hits'] = token_tracks[track_id]['hits']
        enriched['previous_y'] = previous_y
        tracked_detections.append(enriched)

    for track_id in list(unmatched_track_ids):
        track = token_tracks.get(track_id)
        if track is None:
            continue
        track['misses'] += 1
        if (
            track['misses'] > TOKEN_TRACK_MAX_MISSES
            or now - track['last_seen'] > TOKEN_TRACK_MAX_AGE_SECONDS
        ):
            token_tracks.pop(track_id, None)

    return tracked_detections

def draw_detected_tokens(frame, tokens):
    """Draw detected tokens, temporal hit counts and the road ROI."""
    display_frame = frame.copy()
    frame_h, frame_w = display_frame.shape[:2]

    text_colors = {
        'green': (0, 255, 0),
        'yellow': (0, 255, 255),
        'red': (0, 0, 255),
        'hidden': (255, 255, 255)
    }

    if SHOW_ROAD_ROI:
        roi_points = np.array([
            [int(frame_w * ROAD_ROI_TOP_LEFT_X_RATIO), int(frame_h * ROAD_ROI_TOP_Y_RATIO)],
            [int(frame_w * ROAD_ROI_TOP_RIGHT_X_RATIO), int(frame_h * ROAD_ROI_TOP_Y_RATIO)],
            [int(frame_w * ROAD_ROI_BOTTOM_RIGHT_X_RATIO), int(frame_h * ROAD_ROI_BOTTOM_Y_RATIO)],
            [int(frame_w * ROAD_ROI_BOTTOM_LEFT_X_RATIO), int(frame_h * ROAD_ROI_BOTTOM_Y_RATIO)]
        ], dtype=np.int32)
        cv2.polylines(display_frame, [roi_points], True, (255, 255, 255), 1)

    overlay = display_frame.copy()
    for token in tokens:
        visible_color = get_visible_color(token)
        color = text_colors.get(visible_color, (255, 255, 255))
        center = (token['x'], token['y'])
        radius = token['radius']

        cv2.circle(overlay, center, radius, color, -1)

        x1 = max(center[0] - radius, 0)
        y1 = max(center[1] - radius, 0)
        x2 = min(center[0] + radius, frame_w - 1)
        y2 = min(center[1] + radius, frame_h - 1)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

        label = visible_color.upper()
        if 'track_id' in token:
            label += f" T{token['track_id']} H{token.get('hits', 1)}"
        cv2.putText(
            display_frame, label, (x1, max(y1 - 10, 18)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2
        )

    cv2.addWeighted(overlay, 0.30, display_frame, 0.70, 0, display_frame)
    return display_frame

def lane_from_x(x, y=None, frame_width=640, frame_height=480):
    """Return the token's lane OFFSET relative to the front-camera centre.

    Important: this is not an absolute road lane. The camera follows the car,
    so a token directly ahead remains offset 0 even when the car is physically
    in lane -2, -1, +1 or +2. Use absolute_lane_for_token() in the planner.
    """
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


def clamp_lane(lane):
    """Clamp one absolute road-lane index to the valid five-lane range."""
    return max(-2, min(2, int(lane)))


def relative_lane_for_token(token, frame_width, frame_height):
    """Return a token's lane offset relative to the vehicle-mounted camera."""
    return lane_from_x(
        token['x'], token['y'],
        frame_width=frame_width,
        frame_height=frame_height
    )


def absolute_lane_for_token(token, current_vehicle_lane, frame_width, frame_height):
    """Convert a camera-relative token offset into an absolute road lane.

    This conversion fixes the main control bug visible in the recorded run:
    the previous planner compared relative token offsets with absolute vehicle
    lanes, causing it to continue steering toward red and yellow tokens.
    """
    relative_lane = relative_lane_for_token(token, frame_width, frame_height)
    return clamp_lane(current_vehicle_lane + relative_lane)


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
    relative_lane = relative_lane_for_token(token, frame_width, frame_height)
    lane = absolute_lane_for_token(
        token, current_vehicle_lane, frame_width, frame_height
    )
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
        'relative_lane': relative_lane,
        'lane_gap': lane_gap,
        'color': token.get('color'),
        'y': token['y'],
        'reachable': reachable,
        'route_score': float(route_score),
        **metrics
    }



def lane_has_imminent_hazard(lane, tokens, current_vehicle_lane,
                             frame_width, frame_height,
                             minimum_progress=UNWANTED_TOKEN_AVOID_Y_RATIO):
    """Return True if a visible red, yellow or hidden token threatens a lane."""
    for token in tokens:
        color = get_visible_color(token)
        if color not in ['yellow', 'red', 'hidden']:
            continue

        token_lane = absolute_lane_for_token(
            token, current_vehicle_lane, frame_width, frame_height
        )
        progress = token['y'] / max(float(frame_height), 1.0)
        if token_lane == lane and progress >= minimum_progress:
            return True

    return False


def lane_has_blocking_hazard(target_candidate, tokens, current_vehicle_lane,
                             frame_width, frame_height):
    """
    Apply a hard veto if a visible hazard blocks the route to a green target.

    The corridor includes intermediate absolute lanes because the car may need
    to cross them before reaching the intended green-token lane.
    """
    target_lane = target_candidate['lane']
    target_y = target_candidate['y']
    margin = frame_height * HAZARD_BLOCK_Y_MARGIN_RATIO
    corridor_lanes = set(range(
        min(current_vehicle_lane, target_lane),
        max(current_vehicle_lane, target_lane) + 1
    ))

    for token in tokens:
        color = get_visible_color(token)
        if color not in ['yellow', 'red', 'hidden']:
            continue

        token_lane = absolute_lane_for_token(
            token, current_vehicle_lane, frame_width, frame_height
        )
        if token_lane not in corridor_lanes:
            continue

        # Larger y values are closer to the car. Reject routes where the hazard
        # is already in front of, or close behind, the intended green token.
        if token['y'] >= target_y - margin:
            return True

    return False


def choose_emergency_escape_lane(tokens, current_vehicle_lane,
                                 frame_width, frame_height):
    """Choose one adjacent escape lane when the current lane is dangerous.

    An emergency manoeuvre is limited to one lane at a time. This avoids a
    risky multi-lane sweep through a token row while still leaving the current
    hazardous lane as early as possible.
    """
    candidate_lanes = [
        lane for lane in (
            current_vehicle_lane - 1,
            current_vehicle_lane + 1,
            current_vehicle_lane
        )
        if -2 <= lane <= 2
    ]
    return choose_low_risk_lane(
        tokens, current_vehicle_lane, frame_width, frame_height,
        candidate_lanes=candidate_lanes
    )


def choose_low_risk_lane(tokens, current_vehicle_lane, frame_width, frame_height,
                         candidate_lanes=None):
    """Choose a safe lane when no confirmed and reachable green route exists."""
    if candidate_lanes is None:
        candidate_lanes = list(range(-2, 3))

    best_lane = current_vehicle_lane
    best_score = -float('inf')

    for lane in candidate_lanes:
        lane = max(-2, min(2, lane))
        score = -abs(lane - current_vehicle_lane) * LANE_SWITCH_PENALTY

        for token in tokens:
            color = get_visible_color(token)
            if color not in ['green', 'yellow', 'red', 'hidden']:
                continue

            token_lane = absolute_lane_for_token(
                token, current_vehicle_lane, frame_width, frame_height
            )
            if token_lane != lane:
                continue

            progress = token['y'] / max(float(frame_height), 1.0)
            proximity_weight = 1.0 + 4.0 * progress * progress

            if color == 'green' and token.get('hits', 1) >= GREEN_CONFIRM_FRAMES:
                score += GREEN_LANE_REWARD * proximity_weight
            elif progress >= UNWANTED_TOKEN_AVOID_Y_RATIO:
                if color == 'yellow':
                    score -= YELLOW_HAZARD_PENALTY * proximity_weight
                elif color == 'red':
                    score -= RED_HAZARD_PENALTY * proximity_weight
                elif color == 'hidden':
                    score -= UNKNOWN_HAZARD_PENALTY * proximity_weight

        if score > best_score:
            best_score = score
            best_lane = lane

    return best_lane

def select_best_green_candidate(candidates, tokens, current_vehicle_lane,
                                frame_width, frame_height):
    """Select the best confirmed green route using only currently visible types."""
    if not candidates:
        return None

    lane_scores = {}
    for lane in range(-2, 3):
        score = -abs(lane - current_vehicle_lane) * LANE_SWITCH_PENALTY
        for token in tokens:
            color = get_visible_color(token)
            if color not in ['green', 'yellow', 'red', 'hidden']:
                continue

            token_lane = absolute_lane_for_token(
                token, current_vehicle_lane, frame_width, frame_height
            )
            if token_lane != lane:
                continue

            progress = token['y'] / max(float(frame_height), 1.0)
            proximity_weight = 1.0 + 4.0 * progress * progress
            if color == 'green' and token.get('hits', 1) >= GREEN_CONFIRM_FRAMES:
                score += GREEN_LANE_REWARD * proximity_weight
            elif progress >= UNWANTED_TOKEN_AVOID_Y_RATIO:
                if color == 'yellow':
                    score -= YELLOW_HAZARD_PENALTY * proximity_weight
                elif color == 'red':
                    score -= RED_HAZARD_PENALTY * proximity_weight
                elif color == 'hidden':
                    score -= UNKNOWN_HAZARD_PENALTY * proximity_weight
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
    """
    Estimate collection only when a tracked token crosses the pickup line.

    Track IDs avoid double counting a persistent contour and avoid suppressing a
    genuinely new token that appears shortly afterwards in the same lane.
    """
    global locked_green_lane, locked_green_until

    now = time.time()
    collection_y = frame_height * TOKEN_COLLECTION_Y_RATIO
    collected = None

    for token in tokens:
        previous_y = token.get('previous_y', token['y'])
        crossed_collection_line = previous_y < collection_y <= token['y']
        if not crossed_collection_line:
            continue

        relative_lane = relative_lane_for_token(token, frame_width, frame_height)
        if relative_lane != 0:
            continue

        color = get_actual_color(token)
        if color not in ['green', 'yellow', 'red']:
            continue

        token_id = f"track:{token.get('track_id', relative_lane)}"
        with data_lock:
            recent_collections = shared_data.get('last_collected_by_lane_color', {})
            already_counted = token_id in recent_collections

        if already_counted:
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
    """
    Detect tokens from each new front-camera frame and display both previews.

    The back-camera frame remains a live preview only. Tracking is updated only
    once per new front frame so hit counts represent real observations rather
    than repeated processing-loop iterations.
    """
    global last_processed_front_seq

    with data_lock:
        front_frame = shared_data.get('latest_front_frame')
        back_frame = shared_data.get('latest_back_frame')
        front_frame_seq = shared_data.get('front_frame_seq', 0)

    window_updated = False

    if back_frame is not None:
        try:
            back_debug_frame = back_frame.copy()
            cv2.putText(
                back_debug_frame, 'Back Camera Preview', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.imshow('Back Camera', cv2.resize(back_debug_frame, (640, 480)))
            window_updated = True
        except Exception as e:
            print(f'Back camera display error: {e}')

    if front_frame is not None and front_frame_seq != last_processed_front_seq:
        try:
            last_processed_front_seq = front_frame_seq
            processed_front_frame = apply_camera_effects(front_frame)

            tokens = detect_colored_tokens(processed_front_frame)
            tokens = apply_token_visibility_effects(tokens)
            tokens = update_token_tracks(
                tokens,
                processed_front_frame.shape[1],
                processed_front_frame.shape[0]
            )

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
                    debug_frame, f'Yellow Effect: {effect}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                )

            cv2.imshow('Front Camera', cv2.resize(debug_frame, (640, 480)))
            window_updated = True
        except Exception as e:
            print(f'Front camera processing error: {e}')

    if window_updated:
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
        front_frame_time = shared_data.get('front_frame_time', 0.0)
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

    # Do not chase stale perception after a camera interruption.
    if (
        front_frame is None
        or front_frame_time <= 0.0
        or time.monotonic() - front_frame_time > STALE_FRONT_FRAME_SECONDS
    ):
        with data_lock:
            shared_data['decision_debug'] = 'Reason: STALE_FRONT_FRAME | Hold lane safely'
            shared_data['target_token_debug'] = None
        try:
            send_control_packet(0.0, 0.30)
        except Exception:
            pass
        return

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
        # Lane planning must use the currently visible type only. A token whose
        # type was hidden by a yellow effect is treated as an unknown hazard.
        color = get_visible_color(token)
        if color not in ['green', 'yellow', 'red', 'hidden']:
            continue

        candidate = evaluate_token_candidate(
            token, current_lane, img_w, img_h
        )
        candidate['color'] = color
        evaluated_tokens.append(candidate)

        if (
            color == 'green'
            and token.get('hits', 1) >= GREEN_CONFIRM_FRAMES
            and token['y'] >= TOKEN_DECISION_Y_MIN
            and candidate['reachable']
        ):
            green_candidates.append(candidate)

    # A green reward never overrides a red, yellow or unknown token blocking
    # the driving corridor. This is a hard safety veto rather than a soft score.
    green_candidates = [
        candidate for candidate in green_candidates
        if not lane_has_blocking_hazard(
            candidate, tokens_snapshot, current_lane, img_w, img_h
        )
    ]

    # Yellow-token effects can make perception or actuation unreliable. Reduce
    # speed while an effect is active so the controller has more reaction time.
    active_yellow_effect = get_active_yellow_effect()
    if active_yellow_effect is not None:
        acceleration_input = min(acceleration_input, YELLOW_EFFECT_SAFE_ACCELERATION)

    # Safety always overrides collection. Start leaving a hazardous lane before
    # the token reaches the old late threshold, even if a green is also visible.
    current_lane_hazard = lane_has_imminent_hazard(
        current_lane,
        tokens_snapshot,
        current_lane,
        img_w,
        img_h,
        minimum_progress=UNWANTED_TOKEN_AVOID_Y_RATIO
    )
    current_lane_emergency = lane_has_imminent_hazard(
        current_lane,
        tokens_snapshot,
        current_lane,
        img_w,
        img_h,
        minimum_progress=EMERGENCY_HAZARD_Y_RATIO
    )

    chosen_target_lane = None
    active_green_candidate = None
    decision_reason = 'MAINTAIN_SAFE_LANE'
    now = time.time()

    # A visible hazard in the current lane overrides green collection and stale
    # green memory. The emergency branch chooses only one adjacent move.
    if current_lane_hazard:
        chosen_target_lane = choose_emergency_escape_lane(
            tokens_snapshot, current_lane, img_w, img_h
        )
        locked_green_lane = None
        locked_green_until = 0.0
        last_token_lane = None
        last_token_time = 0.0
        acceleration_input = min(
            acceleration_input,
            EMERGENCY_BRAKE_ACCELERATION
            if current_lane_emergency
            else EMERGENCY_ESCAPE_ACCELERATION
        )
        decision_reason = (
            'EMERGENCY_ESCAPE'
            if chosen_target_lane != current_lane
            else 'BRAKE_FOR_HAZARD'
        )

    # Retain a short lock to prevent rapid lane oscillation, but allow a new
    # visible green to replace a stale lock immediately.
    if chosen_target_lane is None and locked_green_lane is not None and now < locked_green_until:
        same_lane_greens = [
            candidate for candidate in green_candidates
            if candidate['lane'] == locked_green_lane
        ]
        active_green_candidate = select_best_green_candidate(
            same_lane_greens, tokens_snapshot, current_lane, img_w, img_h
        )

    if chosen_target_lane is None and active_green_candidate is None:
        active_green_candidate = select_best_green_candidate(
            green_candidates, tokens_snapshot, current_lane, img_w, img_h
        )

    if chosen_target_lane is None and active_green_candidate is not None:
        chosen_target_lane = active_green_candidate['lane']
        locked_green_lane = chosen_target_lane
        locked_green_until = now + GREEN_TARGET_LOCK_SECONDS
        last_token_lane = chosen_target_lane
        last_token_time = now
        decision_reason = 'COLLECT_GREEN'
    elif chosen_target_lane is None and (
        last_token_lane is not None
        and now - last_token_time < TOKEN_TARGET_MEMORY_SECONDS
        and not lane_has_imminent_hazard(
            last_token_lane, tokens_snapshot, current_lane, img_w, img_h
        )
    ):
        # Brief memory prevents a lane change from being aborted by one missed
        # contour, but a newly visible hazard always cancels stale green memory.
        chosen_target_lane = last_token_lane
        decision_reason = 'GREEN_MEMORY'
    elif chosen_target_lane is None:
        chosen_target_lane = choose_low_risk_lane(
            tokens_snapshot, current_lane, img_w, img_h
        )
        decision_reason = (
            'AVOID_RED_YELLOW'
            if chosen_target_lane != current_lane
            else 'MAINTAIN_SAFE_LANE'
        )

    chosen_target_lane = clamp_lane(chosen_target_lane)
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
            f" | Rel: {active_green_candidate.get('relative_lane', 0):+d}"
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
    token_tracks.clear()
    next_token_track_id = 1
    last_processed_front_seq = -1
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
