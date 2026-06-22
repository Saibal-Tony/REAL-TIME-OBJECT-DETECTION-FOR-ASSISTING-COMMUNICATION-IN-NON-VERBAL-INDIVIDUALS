import cv2
import numpy as np
import mediapipe as mp
import os

# ── Config ───────────────────────────────────────
GESTURES        = ['J', 'Z']
SEQUENCES       = 40
SEQUENCE_LENGTH = 60
DATA_PATH       = r'E:\Study\Projects\REAL-TIME OBJECT DETECTION FOR ASSISTING COMMUNICATION IN NON-VERBAL INDIVIDUALS\JZ_Data'

# ── MediaPipe setup ──────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_landmarks(results):
    if results.multi_hand_landmarks:
        return np.array([[lm.x, lm.y, lm.z]
            for lm in results.multi_hand_landmarks[0].landmark]).flatten()
    return np.zeros(63)

def flush_camera(cap, n=3):
    """Discard n buffered frames to get the freshest frame."""
    for _ in range(n):
        cap.grab()

def read_fresh(cap):
    """Always read the freshest frame by flushing buffer first."""
    flush_camera(cap)
    return cap.read()

# ── Create folders ────────────────────────────────
for gesture in GESTURES:
    for seq in range(SEQUENCES):
        os.makedirs(os.path.join(DATA_PATH, gesture, str(seq)), exist_ok=True)

# ── Camera setup ──────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # ← key fix: only keep 1 frame in buffer

if not cap.isOpened():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print('Camera not found!'); exit()

print('Camera ready! Press Q to stop early')
stopped = False

for gesture in GESTURES:
    if stopped: break
    print(f'\n--- Gesture: {gesture} ---')

    for seq in range(SEQUENCES):
        if stopped: break

        seq_path = os.path.join(DATA_PATH, gesture, str(seq))
        existing = len([f for f in os.listdir(seq_path) if f.endswith('.npy')])
        if existing == SEQUENCE_LENGTH:
            print(f'  Seq {seq+1}: already complete, skipping')
            continue

        print(f'  Seq {seq+1}/{SEQUENCES} — GET READY')

        # ── Countdown ────────────────────────────
        for countdown in range(3, 0, -1):
            ret, frame = read_fresh(cap)
            if not ret: continue
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (0,0),
                (frame.shape[1], frame.shape[0]), (0,0,0), -1)
            cv2.putText(frame, str(countdown),
                (frame.shape[1]//2-60, frame.shape[0]//2+50),
                cv2.FONT_HERSHEY_SIMPLEX, 6, (0,255,0), 8)
            cv2.putText(frame, f'Draw: {gesture}',
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,255,0), 4)
            cv2.putText(frame, 'KEEP PALM FACING CAMERA',
                (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2)
            cv2.imshow('Collecting J Z', frame)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                stopped = True; break

        if stopped: break

        # ── Record frames ─────────────────────────
        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = read_fresh(cap)  # fresh frame, no buffer buildup
            if not ret: continue
            frame = cv2.flip(frame, 1)

            # MediaPipe
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False          # small speed boost
            results = hands.process(rgb)
            rgb.flags.writeable = True

            # Draw landmarks
            if results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS)

            # Save landmarks
            lm = extract_landmarks(results)
            np.save(os.path.join(seq_path, str(frame_num)), lm)

            # Progress bar
            prog = int(frame_num / SEQUENCE_LENGTH * frame.shape[1])
            cv2.rectangle(frame,
                (0, frame.shape[0]-20), (prog, frame.shape[0]),
                (0,255,0), -1)

            # Status text
            cv2.putText(frame, f'RECORDING {gesture}',
                (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
            cv2.putText(frame,
                f'Seq {seq+1}/{SEQUENCES}  Frame {frame_num+1}/{SEQUENCE_LENGTH}',
                (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            status = 'Hand OK' if results.multi_hand_landmarks \
                     else 'NO HAND — keep palm visible!'
            color  = (0,255,0) if results.multi_hand_landmarks else (0,0,255)
            cv2.putText(frame, status,
                (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv2.imshow('Collecting J Z', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stopped = True; break

        print(f'  Seq {seq+1} done')

cap.release()
cv2.destroyAllWindows()
print('\n=== DONE ===')
for g in GESTURES:
    d    = os.path.join(DATA_PATH, g)
    seqs = [s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))]
    done = sum(1 for s in seqs
               if len(os.listdir(os.path.join(d, s))) == SEQUENCE_LENGTH)
    print(f'  {g}: {done}/{SEQUENCES} sequences complete')