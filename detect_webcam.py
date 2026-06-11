# import cv2
# import numpy as np
# import tensorflow as tf
# import os
# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as viz_utils

# # Paths
# OUTPUT_PATH = r'E:\Study\Projects\REAL-TIME OBJECT DETECTION FOR ASSISTING COMMUNICATION IN NON-VERBAL INDIVIDUALS\Tensorflow\workspace\models\my_ssd_mobnet_initial\export\saved_model'
# LABELMAP    = r'E:\Study\Projects\REAL-TIME OBJECT DETECTION FOR ASSISTING COMMUNICATION IN NON-VERBAL INDIVIDUALS\Tensorflow\workspace\annotations\label_map.pbtxt'

# # Fix cuDNN
# cudnn_path = r'E:\Study\Projects\REAL-TIME OBJECT DETECTION FOR ASSISTING COMMUNICATION IN NON-VERBAL INDIVIDUALS\ObjEnv\Lib\site-packages\nvidia\cudnn\bin'
# os.environ['PATH'] = cudnn_path + ';' + os.environ.get('PATH', '')
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

# # Load model
# print('Loading model...')
# detect_fn      = tf.saved_model.load(OUTPUT_PATH)
# category_index = label_map_util.create_category_index_from_labelmap(LABELMAP)
# print('✅ Model loaded! Press Q to quit')

# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     image_np     = np.array(frame)
#     input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)
#     detections   = detect_fn(input_tensor)
#     num_det      = int(detections.pop('num_detections'))
#     detections   = {k: v[0, :num_det].numpy() for k, v in detections.items()}
#     detections['num_detections']    = num_det
#     detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

#     viz_utils.visualize_boxes_and_labels_on_image_array(
#         image_np,
#         detections['detection_boxes'],
#         detections['detection_classes'],
#         detections['detection_scores'],
#         category_index,
#         use_normalized_coordinates=True,
#         max_boxes_to_draw=3,
#         min_score_thresh=0.7,
#         agnostic_mode=False)

#     cv2.imshow('Sign Language Detection — Press Q to quit', image_np)

#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO('sign_lang_best.pt')

cap = cv2.VideoCapture(0)
print('✅ Webcam started! Press Q to quit')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, conf=0.7, verbose=False)

    # Draw boxes automatically
    annotated = results[0].plot()

    # Show top detection
    if results[0].boxes:
        top = results[0].boxes[0]
        cls_id = int(top.cls[0])
        conf   = float(top.conf[0])
        label  = model.names[cls_id]
        print(f'Detected: {label} ({conf:.0%})', end='\r')

    cv2.imshow('Sign Language Detection — Press Q to quit', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()