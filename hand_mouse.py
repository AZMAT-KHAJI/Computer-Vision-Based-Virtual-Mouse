import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Camera
cap = cv2.VideoCapture(0)

# Screen size
screen_width, screen_height = pyautogui.size()

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Smooth cursor
prev_x, prev_y = 0, 0
smoothening = 6

# Timing
click_time = 0
double_click_time = 0
dragging = False

# FPS
p_time = 0

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Landmarks
            index = hand_landmarks.landmark[8]
            middle = hand_landmarks.landmark[12]
            thumb = hand_landmarks.landmark[4]

            x = int(index.x * w)
            y = int(index.y * h)

            screen_x = index.x * screen_width
            screen_y = index.y * screen_height

            # -------------------------
            # Finger detection
            # -------------------------
            fingers = []

            if thumb.x < hand_landmarks.landmark[3].x:
                fingers.append(1)
            else:
                fingers.append(0)

            tip_ids = [8, 12, 16, 20]
            for i in range(4):
                if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i]-2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # -------------------------
            # Cursor Movement
            # -------------------------
            if fingers[1] == 1 and fingers[2] == 0:
                curr_x = screen_x
                curr_y = screen_y

                smooth_x = prev_x + (curr_x - prev_x) / smoothening
                smooth_y = prev_y + (curr_y - prev_y) / smoothening

                pyautogui.moveTo(smooth_x, smooth_y)

                prev_x, prev_y = smooth_x, smooth_y

                cv2.putText(frame, "MOVE", (20, 50), 0, 1, (255,255,255), 2)

            # -------------------------
            # Distance for gestures
            # -------------------------
            thumb_x = int(thumb.x * w)
            thumb_y = int(thumb.y * h)
            distance = math.hypot(x - thumb_x, y - thumb_y)

            # -------------------------
            # LEFT CLICK
            # -------------------------
            if distance < 30 and fingers[2] == 0:
                if time.time() - click_time > 1:
                    pyautogui.click()
                    click_time = time.time()

                cv2.putText(frame, "LEFT CLICK", (20, 100), 0, 1, (0,255,0), 2)

            # -------------------------
            # RIGHT CLICK
            # -------------------------
            if fingers[1] == 1 and fingers[2] == 1 and distance < 40:
                pyautogui.rightClick()
                cv2.putText(frame, "RIGHT CLICK", (20, 150), 0, 1, (255,0,0), 2)
                time.sleep(1)

            # -------------------------
            # DOUBLE CLICK
            # -------------------------
            if distance < 20:
                if time.time() - double_click_time < 0.5:
                    pyautogui.doubleClick()
                double_click_time = time.time()

            # -------------------------
            # DRAG & DROP
            # -------------------------
            if distance < 25 and not dragging:
                pyautogui.mouseDown()
                dragging = True

            if distance > 40 and dragging:
                pyautogui.mouseUp()
                dragging = False

            if dragging:
                cv2.putText(frame, "DRAG", (20, 200), 0, 1, (0,255,255), 2)

            # -------------------------
            # SCROLL
            # -------------------------
            if fingers[1] == 1 and fingers[2] == 1 and distance > 50:
                if index.y < 0.5:
                    pyautogui.scroll(80)
                else:
                    pyautogui.scroll(-80)

                cv2.putText(frame, "SCROLL", (20, 250), 0, 1, (255,255,0), 2)

            cv2.circle(frame, (x, y), 10, (255, 0, 255), -1)

    # -------------------------
    # FPS Display
    # -------------------------
    c_time = time.time()
    fps = 1 / (c_time - p_time) if (c_time - p_time) != 0 else 0
    p_time = c_time

    cv2.putText(frame, f'FPS: {int(fps)}', (500, 50), 0, 1, (0,255,0), 2)

    cv2.imshow("AI Gesture Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp
# import pyautogui
# import numpy as np
# import pandas as pd
# import pickle
# import time

# # Load model
# model = pickle.load(open("gesture_knn.pkl", "rb"))

# # Camera
# cap = cv2.VideoCapture(0)

# # Screen size
# screen_width, screen_height = pyautogui.size()

# # Mediapipe
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=1)
# mp_draw = mp.solutions.drawing_utils

# # Smooth movement
# prev_x, prev_y = 0, 0
# smoothening = 6

# # Click delay
# last_click_time = 0
# click_delay = 1

# # Scroll delay
# last_scroll_time = 0
# scroll_delay = 0.3

# while True:
#     success, frame = cap.read()
#     if not success:
#         break

#     frame = cv2.flip(frame, 1)
#     h, w, _ = frame.shape

#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(rgb)

#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:

#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             # Index finger
#             index = hand_landmarks.landmark[8]

#             x = int(index.x * w)
#             y = int(index.y * h)

#             screen_x = index.x * screen_width
#             screen_y = index.y * screen_height

#             # -------------------------
#             # FEATURE EXTRACTION
#             # -------------------------
#             features = []
#             for lm in hand_landmarks.landmark:
#                 features.append(lm.x)
#                 features.append(lm.y)

#             features_df = pd.DataFrame([features])

#             # -------------------------
#             # PREDICTION
#             # -------------------------
#             prediction = model.predict(features_df)[0]

#             # -------------------------
#             # ACTIONS
#             # -------------------------

#             # MOVE
#             if prediction == "move":
#                 curr_x = screen_x
#                 curr_y = screen_y

#                 smooth_x = prev_x + (curr_x - prev_x) / smoothening
#                 smooth_y = prev_y + (curr_y - prev_y) / smoothening

#                 pyautogui.moveTo(smooth_x, smooth_y)
#                 prev_x, prev_y = smooth_x, smooth_y

#                 cv2.putText(frame, "MOVE", (20, 50), 0, 1, (255,255,255), 2)

#             # CLICK
#             elif prediction == "click":
#                 if time.time() - last_click_time > click_delay:
#                     pyautogui.click()
#                     last_click_time = time.time()

#                 cv2.putText(frame, "CLICK", (20, 100), 0, 1, (0,255,0), 2)

#             # SCROLL
#             elif prediction == "scroll":
#                 if time.time() - last_scroll_time > scroll_delay:
#                     if index.y < 0.5:
#                         pyautogui.scroll(80)
#                     else:
#                         pyautogui.scroll(-80)

#                     last_scroll_time = time.time()

#                 cv2.putText(frame, "SCROLL", (20, 150), 0, 1, (255,255,0), 2)

#             # POINTER
#             cv2.circle(frame, (x, y), 10, (255, 0, 255), -1)

#             # SHOW PREDICTION
#             cv2.putText(frame, f"Pred: {prediction}", (20, 200),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

#     cv2.imshow("AI Gesture Mouse (KNN)", frame)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()