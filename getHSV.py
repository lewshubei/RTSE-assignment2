import cv2
import numpy as np

# --- Load your image or frame ---
frame = cv2.imread("redToken.png")  # Replace with your screenshot
# OR if you want to get from simulator:
# from sample_drive import shared_data, data_lock
# with data_lock:
#     frame = shared_data['latest_front_frame'].copy()

# --- Mouse callback to print HSV ---
def get_hsv_on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param
        pixel_bgr = frame[y, x]
        pixel_hsv = cv2.cvtColor(pixel_bgr.reshape(1,1,3), cv2.COLOR_BGR2HSV)[0][0]
        print(f"Clicked at ({x}, {y}) → BGR: {pixel_bgr}, HSV: {pixel_hsv}")

cv2.namedWindow("Calibration Frame")
cv2.setMouseCallback("Calibration Frame", get_hsv_on_click, param=frame)

print("Click on red tokens to get HSV values. Press 'q' to quit.")

while True:
    cv2.imshow("Calibration Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press q to exit safely
        break

cv2.destroyAllWindows()