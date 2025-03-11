import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import math
import vgamepad as vg  # Requires Windows and a supported gamepad emulator

# -------------------------
# Global Variables & Setup
# -------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Unable to open webcam")

global_frame = None
frame_lock = threading.Lock()

# Global flag to control the frame_reader thread.
running = True

# State variables for button presses.
a_pressed = False
b_pressed = False

def frame_reader():
    global global_frame, running
    while running:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                global_frame = frame.copy()
        time.sleep(0.01)

threading.Thread(target=frame_reader, daemon=True).start()

# Initialize MediaPipe Hands with support for two hands.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Allow both left and right hands.
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Initialize virtual gamepad.
gamepad = vg.VX360Gamepad()

# -------------------------
# Helper Functions
# -------------------------

def draw_mirrored_text(image, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX,
                         font_scale=0.7, color=(255, 0, 0), thickness=2):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_img = np.zeros((text_height + baseline, text_width, 3), dtype=np.uint8)
    cv2.putText(text_img, text, (0, text_height), font, font_scale, color, thickness, cv2.LINE_AA)
    mirrored_text_img = cv2.flip(text_img, 1)
    gray = cv2.cvtColor(mirrored_text_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    x, y = pos
    h_img, w_img, _ = image.shape
    if x + text_width <= w_img and y + text_height + baseline <= h_img:
        roi = image[y:y + text_height + baseline, x:x + text_width]
        roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        text_fg = cv2.bitwise_and(mirrored_text_img, mirrored_text_img, mask=mask)
        dst = cv2.add(roi_bg, text_fg)
        image[y:y + text_height + baseline, x:x + text_width] = dst
    return image

def update_controller_stick(norm_x, norm_y):
    """
    Convert continuous joystick values (-1 to 1) to the VX360 gamepad axis values.
    The left joystick axis ranges from 0 to 32767 with center at 16384.
    """
    x_val = int(2 * (16384 - (16384 + norm_x * 16383)))
    y_val = int(-2 * (16384 - (16384 - norm_y * 16383)))  # Invert y to match controller's convention
    gamepad.left_joystick(x_value=x_val, y_value=y_val)
    # Note: gamepad.update() is called once per frame after all state changes.

def process_frame(frame):
    global a_pressed, b_pressed
    proc_frame = cv2.resize(frame, (640, 480))
    image_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    annotated = proc_frame.copy()
    
    joystick_dir = (0, 0)
    left_hand_found = False
    right_hand_found = False
    pinch_threshold = 0.07

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[i].classification[0].label
            h, w, _ = annotated.shape
            
            if handedness == "Right":
                left_hand_found = True
                # Use landmarks 5 and 8 for joystick control.
                lm5 = hand_landmarks.landmark[5]
                lm8 = hand_landmarks.landmark[8]
                x5, y5 = int(lm5.x * w), int(lm5.y * h)
                x8, y8 = int(lm8.x * w), int(lm8.y * h)
                dx = x8 - x5
                dy = y8 - y5

                angle_rad = math.atan2(dy, dx)
                angle_deg = math.degrees(angle_rad)
                angle_text = f"Angle: {int(angle_deg)} deg"
                annotated = draw_mirrored_text(annotated, angle_text, (x5, max(y5 - 30, 0)))
                
                norm_x = math.cos(angle_rad)
                norm_y = math.sin(angle_rad)
                joystick_dir = (norm_x, norm_y)
                
                # Trigger logic for left/right triggers:
                if 1 <= angle_deg <= 180:
                    gamepad.left_trigger(value=255)
                else:
                    gamepad.left_trigger(value=0)
                
                if -180 <= angle_deg <= -1:
                    gamepad.right_trigger(value=255)
                else:
                    gamepad.right_trigger(value=0)
                
                mp_drawing.draw_landmarks(
                    annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                cv2.arrowedLine(annotated, (x5, y5), (x8, y8), (255, 0, 0), 3, tipLength=0.2)
            
            elif handedness == "Left":
                right_hand_found = True
                try:
                    thumb = hand_landmarks.landmark[4]
                    index_finger = hand_landmarks.landmark[8]
                    middle_finger = hand_landmarks.landmark[12]
                    pinch_index_distance = math.hypot(thumb.x - index_finger.x, thumb.y - index_finger.y)
                    pinch_middle_distance = math.hypot(thumb.x - middle_finger.x, thumb.y - middle_finger.y)
                    
                    thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
                    
                    if pinch_index_distance < pinch_threshold:
                        if not a_pressed:
                            try:
                                gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
                                a_pressed = True
                                print("Pressed A")
                            except Exception as e:
                                print("Error pressing A button:", e)
                        cv2.putText(annotated, "A", (thumb_x, thumb_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    else:
                        if a_pressed:
                            try:
                                gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
                                a_pressed = False
                                print("Released A")
                            except Exception as e:
                                print("Error releasing A button:", e)
                    
                    if pinch_middle_distance < pinch_threshold:
                        if not b_pressed:
                            try:
                                gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
                                b_pressed = True
                                print("Pressed B")
                            except Exception as e:
                                print("Error pressing B button:", e)
                        cv2.putText(annotated, "B", (thumb_x, thumb_y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    else:
                        if b_pressed:
                            try:
                                gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
                                b_pressed = False
                                print("Released B")
                            except Exception as e:
                                print("Error releasing B button:", e)
                    
                    mp_drawing.draw_landmarks(
                        annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                    )
                except Exception as e:
                    print("Error processing left hand landmarks:", e)

    if not left_hand_found:
        joystick_dir = (0, 0)
        gamepad.left_trigger(value=0)
        gamepad.right_trigger(value=0)
    if not right_hand_found:
        if a_pressed:
            gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            a_pressed = False
        if b_pressed:
            gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
            b_pressed = False
    
    update_controller_stick(*joystick_dir)
    
    # Call update() once after all state changes.
    gamepad.update()
    
    processed = cv2.flip(annotated, 1)
    return processed

# -------------------------
# Tkinter GUI Setup
# -------------------------

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # Frames for raw and processed video feeds.
        video_frame = ttk.Frame(window)
        video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.raw_label = ttk.Label(video_frame, text="Raw Video")
        self.raw_label.grid(row=0, column=0, padx=5, pady=5)
        
        self.proc_label = ttk.Label(video_frame, text="Processed Video")
        self.proc_label.grid(row=0, column=1, padx=5, pady=5)
        
        # Control buttons.
        control_frame = ttk.Frame(window)
        control_frame.pack(fill=tk.X, expand=True)
        self.start_btn = ttk.Button(control_frame, text="Start", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop)
        self.stop_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.running = False
        self.delay = 30  # milliseconds delay for update loop
        self.update()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()
    
    def start(self):
        self.running = True
        
    def stop(self):
        self.running = False
        
    def update(self):
        if self.running:
            with frame_lock:
                frame = None if global_frame is None else global_frame.copy()
            if frame is not None:
                # Processed feed.
                proc_frame = process_frame(frame)
                proc_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
                proc_img = Image.fromarray(proc_rgb)
                proc_imgtk = ImageTk.PhotoImage(image=proc_img)
                self.proc_label.imgtk = proc_imgtk  # keep reference
                self.proc_label.configure(image=proc_imgtk)
                
                # Raw feed (mirrored for display).
                raw_frame = cv2.resize(frame, (640, 480))
                raw_frame = cv2.flip(raw_frame, 1)
                raw_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                raw_img = Image.fromarray(raw_rgb)
                raw_imgtk = ImageTk.PhotoImage(image=raw_img)
                self.raw_label.imgtk = raw_imgtk  # keep reference
                self.raw_label.configure(image=raw_imgtk)
        self.window.after(self.delay, self.update)
    
    def on_closing(self):
        global running
        running = False
        cap.release()
        self.window.destroy()

if __name__ == '__main__':
    App(tk.Tk(), "Hand Tracking & Virtual Joystick")
