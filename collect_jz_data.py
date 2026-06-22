import cv2
import numpy as np
import mediapipe as mp
import os
import time

# ── Config ───────────────────────────────────────
GESTURES        = ['J', 'Z']
SEQUENCES       = 30   # 30 videos per gesture
SEQUENCE_LENGTH = 30   # 30 frames per video
DATA_PATH       = 'JZ_Data'

# ── MediaPipe setup ──────────────────────────────
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
hands       = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ── Create folders ───────────────────────────────
for gesture in GESTURES:
    for seq in range(SEQUENCES):
        os.makedirs(os.path.join(DATA_PATH, gesture, str(seq)), exist_ok=True)

def extract_landmarks(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z]
                         for lm in hand.landmark]).flatten()  # shape (63,)
    return np.zeros(63)

# ── Collect data ─────────────────────────────────
cap = cv2.VideoCapture(0)

for gesture in GESTURES:
    print(f'\n{"="*40}')
    print(f'Collecting gesture: {gesture}')
    print(f'{"="*40}')

    for seq in range(SEQUENCES):
        print(f'\nSequence {seq+1}/{SEQUENCES}')
        print('Get ready...')

        # Countdown
        for countdown in range(3, 0, -1):
            ret, frame = cap.read()
            cv2.putText(frame, f'GET READY: {countdown}',
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                       2, (0, 255, 0), 3)
            cv2.putText(frame, f'Gesture: {gesture} | Seq: {seq+1}/{SEQUENCES}',
                       (50, 200), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (255, 255, 0), 2)
            cv2.imshow('Collecting JZ Data', frame)
            cv2.waitKey(1000)

        # Collect frames
        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw landmarks
            if results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS
                )

            landmarks = extract_landmarks(results)

            # Save frame landmarks
            npy_path = os.path.join(
                DATA_PATH, gesture, str(seq), str(frame_num))
            np.save(npy_path, landmarks)

            # Display
            cv2.putText(frame, f'RECORDING {gesture}',
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                       1.5, (0, 0, 255), 3)
            cv2.putText(frame,
                       f'Seq: {seq+1}/{SEQUENCES} | Frame: {frame_num+1}/{SEQUENCE_LENGTH}',
                       (50, 120), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (255, 255, 0), 2)
            cv2.imshow('Collecting JZ Data', frame)
            cv2.waitKey(1)

        print(f'  ✅ Sequence {seq+1} collected')

cap.release()
cv2.destroyAllWindows()
print('\n✅ All data collected!')
print(f'   Saved to: {DATA_PATH}/')