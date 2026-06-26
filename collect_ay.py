
import cv2, os, time

RAW_PATH = r"E:\Study\Projects\REAL-TIME OBJECT DETECTION FOR ASSISTING COMMUNICATION IN NON-VERBAL INDIVIDUALS\AY_data\Tensorflow\workspace\images\collectedimages"
CLASSES  = ['C', 'D', 'G']
TARGET   = 20

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

print("Controls: SPACE=capture | N=next class | Q=quit")

class_idx = 0
while class_idx < len(CLASSES):
    current_class = CLASSES[class_idx]
    class_dir = os.path.join(RAW_PATH, current_class)
    existing = len([f for f in os.listdir(class_dir) if f.endswith(".jpg")])
    count = existing

    while count < TARGET:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        cv2.rectangle(display, (0,0), (display.shape[1], 90), (0,0,0), -1)
        cv2.putText(display, f"Class: {current_class}  ({count}/{TARGET})",
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
        cv2.putText(display, "SPACE=capture | N=next | Q=quit",
                   (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.imshow("Collecting A-Y Images", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            fname = f"{current_class}_{int(time.time()*1000)}.jpg"
            cv2.imwrite(os.path.join(class_dir, fname), frame)
            count += 1
            print(f"  {current_class}: {count}/{TARGET} saved")
        elif key == ord("n"):
            break
        elif key == ord("q"):
            cap.release(); cv2.destroyAllWindows(); exit()

    class_idx += 1

cap.release()
cv2.destroyAllWindows()
print("\n✅ Collection done!")
