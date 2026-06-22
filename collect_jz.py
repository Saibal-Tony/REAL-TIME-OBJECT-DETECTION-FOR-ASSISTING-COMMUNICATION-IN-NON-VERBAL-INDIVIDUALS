
import cv2, numpy as np, mediapipe as mp, os, time

DATA_PATH       = r"E:\Study\Projects\REAL-TIME OBJECT DETECTION FOR ASSISTING COMMUNICATION IN NON-VERBAL INDIVIDUALS\JZ_Data"
GESTURES        = ['J', 'Z']
SEQUENCES       = 40
SEQUENCE_LENGTH = 30

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

def extract_landmarks(results):
    if results.multi_hand_landmarks:
        return np.array([[lm.x, lm.y, lm.z]
            for lm in results.multi_hand_landmarks[0].landmark]).flatten()
    return np.zeros(63)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not found! Try camera index 1")
    exit()

print("✅ Camera ready!")
print("   Collecting", SEQUENCES, "sequences per gesture")
print("   Press Q to stop early\n")

stopped = False
for gesture in GESTURES:
    if stopped: break
    print(f"\n--- Gesture: {gesture} ---")
    for seq in range(SEQUENCES):
        if stopped: break
        seq_path = os.path.join(DATA_PATH, gesture, str(seq))
        # Skip already collected
        existing = len([f for f in os.listdir(seq_path) if f.endswith(".npy")])
        if existing == SEQUENCE_LENGTH:
            print(f"  Seq {seq+1}: already exists, skipping")
            continue
        print(f"  Seq {seq+1}/40 — GET READY to draw {gesture}!")
        for countdown in range(3, 0, -1):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0,0,0), -1)
            cv2.putText(frame, str(countdown), (frame.shape[1]//2-60, frame.shape[0]//2+50),
                       cv2.FONT_HERSHEY_SIMPLEX, 6, (0,255,0), 8)
            cv2.putText(frame, f"Draw: {gesture}", (50, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,255,0), 4)
            cv2.putText(frame, f"Seq {seq+1}/40", (50, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200,200,200), 2)
            cv2.imshow("Collecting J Z Data — Press Q to stop", frame)
            if cv2.waitKey(1000) & 0xFF == ord("q"): stopped=True; break
        if stopped: break
        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res  = hands.process(rgb)
            if res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            lm = extract_landmarks(res)
            np.save(os.path.join(seq_path, str(frame_num)), lm)
            prog = int(frame_num / SEQUENCE_LENGTH * frame.shape[1])
            cv2.rectangle(frame, (0, frame.shape[0]-20), (prog, frame.shape[0]), (0,255,0), -1)
            cv2.putText(frame, f"RECORDING {gesture}", (50,70),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
            cv2.putText(frame, f"Frame {frame_num+1}/30", (50,140),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
            hand_status = "✓ Hand detected" if res.multi_hand_landmarks else "✗ No hand!"
            cv2.putText(frame, hand_status, (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                       (0,255,0) if res.multi_hand_landmarks else (0,0,255), 2)
            cv2.imshow("Collecting J Z Data — Press Q to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): stopped=True; break
        print(f"  ✅ Seq {seq+1} done")

cap.release()
cv2.destroyAllWindows()
print("\n=== COLLECTION DONE ===")
for g in GESTURES:
    d = os.path.join(DATA_PATH, g)
    seqs = [s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))]
    complete = sum(1 for s in seqs if len(os.listdir(os.path.join(d, s))) == SEQUENCE_LENGTH)
    print(f"  {g}: {complete}/{SEQUENCES} sequences complete")
