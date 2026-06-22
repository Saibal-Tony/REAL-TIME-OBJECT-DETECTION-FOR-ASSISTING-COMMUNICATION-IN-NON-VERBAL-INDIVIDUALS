import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from ultralytics import YOLO
from collections import deque
import time

# ── Load models ───────────────────────────────────
print('Loading models...')
yolo_model = YOLO('sign_lang_best.pt')          # A-I, K-Y
lstm_model = tf.keras.models.load_model('jz_lstm_model.keras')  # J, Z
print('✅ Both models loaded!')

# ── Config ───────────────────────────────────────
YOLO_CLASSES    = list('ABCDEFGHIKLMNOPQRSTUVWXY')
LSTM_CLASSES    = ['J', 'Z']
SEQUENCE_LENGTH = 30
CONF_THRESHOLD  = 0.7

# ── MediaPipe setup ──────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# ── State ─────────────────────────────────────────
landmark_sequence = deque(maxlen=SEQUENCE_LENGTH)
current_letter    = ''
letter_history    = []
detected_word     = ''

def extract_landmarks(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z]
                         for lm in hand.landmark]).flatten()
    return np.zeros(63)

# ── Start webcam ─────────────────────────────────
cap = cv2.VideoCapture(0)
print('✅ Starting combined detection...')
print('   A-Y: Detected by YOLOv8')
print('   J,Z: Detected by LSTM')
print('   Press Q to quit | SPACE to add letter | BACKSPACE to delete')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── YOLO detection (A-Y) ──────────────────────
    yolo_results  = yolo_model(frame, conf=CONF_THRESHOLD, verbose=False)
    yolo_detected = ''
    yolo_conf     = 0

    if yolo_results[0].boxes:
        top       = yolo_results[0].boxes[0]
        yolo_detected = yolo_model.names[int(top.cls[0])]
        yolo_conf     = float(top.conf[0])

    # ── MediaPipe + LSTM detection (J, Z) ─────────
    mp_results  = hands.process(image_rgb)
    landmarks   = extract_landmarks(mp_results)
    landmark_sequence.append(landmarks)

    lstm_detected = ''
    lstm_conf     = 0

    if len(landmark_sequence) == SEQUENCE_LENGTH:
        seq_array = np.expand_dims(
            np.array(landmark_sequence), axis=0)  # (1, 30, 63)
        prediction = lstm_model.predict(seq_array, verbose=0)[0]
        lstm_conf  = float(np.max(prediction))
        if lstm_conf > CONF_THRESHOLD:
            lstm_detected = LSTM_CLASSES[np.argmax(prediction)]

    # ── Draw MediaPipe landmarks ──────────────────
    if mp_results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            mp_results.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS
        )

    # ── Draw YOLO boxes ───────────────────────────
    frame = yolo_results[0].plot() if yolo_results[0].boxes else frame

    # ── Decide final detection ────────────────────
    if yolo_conf > CONF_THRESHOLD and yolo_detected not in ['J', 'Z']:
        current_letter = yolo_detected
        conf_display   = yolo_conf
        source         = 'YOLO'
    elif lstm_conf > CONF_THRESHOLD:
        current_letter = lstm_detected
        conf_display   = lstm_conf
        source         = 'LSTM'
    else:
        conf_display = 0
        source       = ''

    # ── UI Display ───────────────────────────────
    h, w = frame.shape[:2]

    # Background box for info
    cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)

    # Current letter
    if current_letter:
        cv2.putText(frame, current_letter,
                   (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                   3, (0, 255, 0), 4)
        cv2.putText(frame,
                   f'{conf_display:.0%} [{source}]',
                   (120, 60), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (255, 255, 0), 2)

    # Word being built
    cv2.rectangle(frame, (0, h-80), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f'Word: {"".join(letter_history)}',
               (20, h-40), cv2.FONT_HERSHEY_SIMPLEX,
               1.2, (255, 255, 255), 2)
    cv2.putText(frame,
               'SPACE=Add Letter | BACKSPACE=Delete | Q=Quit',
               (20, h-10), cv2.FONT_HERSHEY_SIMPLEX,
               0.5, (150, 150, 150), 1)

    cv2.imshow('Sign Language Detection — Combined', frame)

    # ── Key handling ─────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' ') and current_letter:
        letter_history.append(current_letter)
        print(f'Added: {current_letter} | Word so far: {"".join(letter_history)}')
    elif key == 8 and letter_history:  # backspace
        removed = letter_history.pop()
        print(f'Removed: {removed} | Word so far: {"".join(letter_history)}')

cap.release()
cv2.destroyAllWindows()
print(f'\nFinal word: {"".join(letter_history)}')