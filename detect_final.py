# save as detect_final.py and run: python detect_final.py

import cv2, numpy as np, time, torch, gc
import mediapipe as mp
import tensorflow as tf
from ultralytics import YOLO
from collections import deque
import os

# ── Speed optimizations ───────────────────────────
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Load models ───────────────────────────────────
gc.collect()
torch.cuda.empty_cache()

print('Loading models...')
yolo_model = YOLO('AY_data/models/final_v4/weights/sign_lang_AY_v4.pt')

# Export to ONNX first for faster CPU inference (run once)
# yolo_model.export(format='onnx', imgsz=320)
# Then use: yolo_model = YOLO('...best.onnx')

with tf.device('/CPU:0'):
    lstm_model = tf.keras.models.load_model('./jz_lstm_model.keras')

print('✅ Models loaded!')

# ── Config ────────────────────────────────────────
CONF_THRESHOLD  = 0.6   # slightly higher = fewer wrong detections
GESTURES        = ['J', 'Z']
SEQUENCE_LENGTH = 60
NUM_LANDMARKS   = 63
IMG_SIZE        = 320   # ← reduced from 640! Still accurate, 4x faster

# ── MediaPipe ─────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_det  = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0)   # ← 0 = fastest MediaPipe model

def extract_landmarks(results):
    if results.multi_hand_landmarks:
        return np.array([[lm.x, lm.y, lm.z]
            for lm in results.multi_hand_landmarks[0].landmark]).flatten()
    return np.zeros(NUM_LANDMARKS)

# ── State ─────────────────────────────────────────
sequence       = deque(maxlen=SEQUENCE_LENGTH)
letter_history = []
current_letter = ''
last_letter    = ''
stable_count   = 0
STABLE_NEEDED  = 25
last_add_time  = 0
AUTO_ADD_DELAY = 1.5

# FPS tracking
fps_start = time.time()
fps_count = 0
fps_show  = 0

# ── Camera ────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

# ── LSTM skip counter (run every N frames) ────────
LSTM_EVERY   = 3   # run LSTM every 3 frames
lstm_counter = 0
last_lstm_det, last_lstm_conf = '', 0

print('Running! Press Q to quit | SPACE to add letter | BACKSPACE to delete')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)

    # ── Resize for YOLO (320 instead of 640) ─────
    small = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    # ── YOLO ──────────────────────────────────────
    yolo_res        = yolo_model(small, conf=CONF_THRESHOLD,
                                  verbose=False, imgsz=IMG_SIZE)
    yolo_det, yolo_conf = '', 0
    if yolo_res[0].boxes:
        top       = yolo_res[0].boxes[0]
        yolo_det  = yolo_model.names[int(top.cls[0])]
        yolo_conf = float(top.conf[0])

    # ── MediaPipe ──────────────────────────────────
    rgb              = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    mp_res           = hands_det.process(rgb)
    rgb.flags.writeable = True

    if mp_res.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame,
            mp_res.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS)

    sequence.append(extract_landmarks(mp_res))

    # ── LSTM (skip frames for speed) ─────────────
    lstm_counter += 1
    if lstm_counter >= LSTM_EVERY and len(sequence) == SEQUENCE_LENGTH:
        lstm_counter = 0
        with tf.device('/CPU:0'):
            pred = lstm_model.predict(
                np.expand_dims(np.array(sequence), 0), verbose=0)[0]
        last_lstm_conf = float(np.max(pred))
        if last_lstm_conf > CONF_THRESHOLD:
            last_lstm_det = GESTURES[np.argmax(pred)]
        else:
            last_lstm_det = ''

    # ── Final decision ────────────────────────────
    if yolo_conf >= CONF_THRESHOLD and yolo_det not in ['J', 'Z']:
        current_letter = yolo_det;     conf = yolo_conf;      src = 'YOLO'
    elif last_lstm_conf >= CONF_THRESHOLD:
        current_letter = last_lstm_det; conf = last_lstm_conf; src = 'LSTM'
    else:
        current_letter = '';            conf = 0;               src = ''

    # ── Auto-add ──────────────────────────────────
    if current_letter and current_letter == last_letter:
        stable_count += 1
        if (stable_count >= STABLE_NEEDED
                and time.time() - last_add_time > AUTO_ADD_DELAY):
            letter_history.append(current_letter)
            last_add_time = time.time()
            stable_count  = 0
            print(f'Added: {current_letter} | Word: {"".join(letter_history)}')
    else:
        stable_count = 0
    last_letter = current_letter

    # ── Draw YOLO box on original frame ──────────
    if yolo_res[0].boxes:
        # Scale box coords from 320 back to 640x480
        sx = frame.shape[1] / IMG_SIZE
        sy = frame.shape[0] / IMG_SIZE
        for box in yolo_res[0].boxes:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            cv2.rectangle(frame,
                (int(x1*sx), int(y1*sy)),
                (int(x2*sx), int(y2*sy)),
                (0, 255, 0), 2)
            cv2.putText(frame,
                f'{yolo_model.names[int(box.cls[0])]} {float(box.conf[0]):.0%}',
                (int(x1*sx), int(y1*sy)-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # ── FPS ───────────────────────────────────────
    fps_count += 1
    if time.time() - fps_start >= 1.0:
        fps_show  = fps_count
        fps_count = 0
        fps_start = time.time()

    # ── UI ────────────────────────────────────────
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0,0), (w,130), (0,0,0), -1)
    cv2.rectangle(frame, (0,h-75), (w,h), (0,0,0), -1)

    if current_letter:
        cv2.putText(frame, current_letter, (15,105),
            cv2.FONT_HERSHEY_SIMPLEX, 4, (0,255,0), 5)
        cv2.putText(frame, f'[{src}] {conf:.0%}',
            (170,65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,0), 2)

    cv2.putText(frame, f'FPS: {fps_show}',
        (w-110,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)

    if stable_count > 0:
        bar = int(stable_count / STABLE_NEEDED * w)
        cv2.rectangle(frame, (0,130), (bar,145), (0,200,100), -1)

    seq_bar = int(len(sequence) / SEQUENCE_LENGTH * w)
    cv2.rectangle(frame, (0,h-5), (seq_bar,h), (0,100,200), -1)

    word = ''.join(letter_history)
    cv2.putText(frame,
        f'Word: {word if word else "..."}',
        (15,h-40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)
    cv2.putText(frame, 'Q=quit | SPACE=add | BKSP=delete',
        (15,h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,150,150), 1)

    cv2.imshow('A-Z Sign Language Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord(' ') and current_letter:
        letter_history.append(current_letter)
        last_add_time = time.time()
        print(f'Added: {current_letter} | Word: {"".join(letter_history)}')
    elif key == 8 and letter_history:
        removed = letter_history.pop()
        print(f'Removed: {removed} | Word: {"".join(letter_history)}')

cap.release()
hands_det.close()
cv2.destroyAllWindows()
print(f'\nFinal word: {"".join(letter_history)}')