"""
Auto-Labeling Script for Sign Language Detection
-------------------------------------------------
- Uses your trained TF model to auto-label unlabeled images
- Generates Pascal VOC .xml files (LabelImg compatible)
- Flags low-confidence images for manual review
"""

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
import tensorflow as tf
from PIL import Image
import json

# ─────────────────────────────────────────────
# CONFIG — edit these paths if needed
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TENSORFLOW_DIR = os.path.join(BASE_DIR, "Tensorflow")

IMAGES_DIR     = os.path.join(TENSORFLOW_DIR, "workspace", "images", "collectedimages")
MODEL_PATH     = os.path.join(TENSORFLOW_DIR, "workspace", "models")   # folder with saved_model
LABEL_MAP_PATH = os.path.join(TENSORFLOW_DIR, "workspace", "annotations", "label_map.pbtxt")

CONFIDENCE_THRESHOLD = 0.5   # boxes below this go to manual review list
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # 26 classes

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
def load_model(model_path):
    print(f"[INFO] Loading model from: {model_path}")
    model = tf.saved_model.load(model_path)
    print("[INFO] Model loaded successfully!")
    return model

# ─────────────────────────────────────────────
# PARSE LABEL MAP
# ─────────────────────────────────────────────
def parse_label_map(label_map_path):
    label_map = {}
    with open(label_map_path, 'r') as f:
        content = f.read()
    import re
    items = re.findall(r"item\s*\{([^}]*)\}", content)
    for item in items:
        id_match   = re.search(r"id\s*:\s*(\d+)", item)
        name_match = re.search(r"name\s*:\s*'([^']*)'", item)
        if id_match and name_match:
            label_map[int(id_match.group(1))] = name_match.group(1)
    print(f"[INFO] Loaded {len(label_map)} classes from label map")
    return label_map

# ─────────────────────────────────────────────
# CREATE PASCAL VOC XML
# ─────────────────────────────────────────────
def create_xml(filename, folder, width, height, detections):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = folder
    ET.SubElement(annotation, "filename").text = filename

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text  = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text  = "3"

    for det in detections:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text       = det["class"]
        ET.SubElement(obj, "pose").text       = "Unspecified"
        ET.SubElement(obj, "truncated").text  = "0"
        ET.SubElement(obj, "difficult").text  = "0"
        ET.SubElement(obj, "confidence").text = str(round(det["score"], 4))

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(det["xmin"])
        ET.SubElement(bndbox, "ymin").text = str(det["ymin"])
        ET.SubElement(bndbox, "xmax").text = str(det["xmax"])
        ET.SubElement(bndbox, "ymax").text = str(det["ymax"])

    xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ")
    return xml_str

# ─────────────────────────────────────────────
# RUN DETECTION ON ONE IMAGE
# ─────────────────────────────────────────────
def detect_image(model, image_path):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    img_np = np.array(img)
    input_tensor = tf.convert_to_tensor(img_np)[tf.newaxis, ...]
    detections = model(input_tensor)
    return detections, width, height

# ─────────────────────────────────────────────
# MAIN AUTO-LABELING LOOP
# ─────────────────────────────────────────────
def auto_label(model_path):
    model     = load_model(model_path)
    label_map = parse_label_map(LABEL_MAP_PATH)
    infer     = model.signatures["serving_default"]

    manual_review = []   # images that need human review
    total_labeled = 0
    total_skipped = 0

    for class_name in CLASSES:
        class_dir = os.path.join(IMAGES_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"[WARN] Folder not found: {class_dir}")
            continue

        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"\n[INFO] Processing class '{class_name}' — {len(images)} images")

        for img_file in images:
            img_path = os.path.join(class_dir, img_file)
            xml_path = os.path.join(class_dir, os.path.splitext(img_file)[0] + ".xml")

            # Skip already labeled images
            if os.path.exists(xml_path):
                print(f"  [SKIP] Already labeled: {img_file}")
                total_skipped += 1
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                width, height = img.size
                img_np = np.array(img)
                input_tensor = tf.convert_to_tensor(img_np, dtype=tf.uint8)[tf.newaxis, ...]

                detections = infer(input_tensor)

                boxes   = detections["detection_boxes"].numpy()[0]
                scores  = detections["detection_scores"].numpy()[0]
                classes = detections["detection_classes"].numpy()[0].astype(int)

                valid_detections = []
                max_score = 0.0

                for box, score, cls in zip(boxes, scores, classes):
                    if score < CONFIDENCE_THRESHOLD:
                        continue
                    if score > max_score:
                        max_score = score

                    ymin, xmin, ymax, xmax = box
                    valid_detections.append({
                        "class": label_map.get(cls, class_name),
                        "score": float(score),
                        "xmin":  int(xmin * width),
                        "ymin":  int(ymin * height),
                        "xmax":  int(xmax * width),
                        "ymax":  int(ymax * height),
                    })

                if not valid_detections:
                    # No confident detection — add to manual review
                    manual_review.append({
                        "image": img_path,
                        "reason": "no confident detection",
                        "max_score": float(max_score)
                    })
                    print(f"  [REVIEW] No confident detection: {img_file} (max score: {max_score:.2f})")
                    continue

                # Save XML
                xml_str = create_xml(img_file, class_name, width, height, valid_detections)
                with open(xml_path, "w") as f:
                    f.write(xml_str)

                total_labeled += 1
                print(f"  [OK] Labeled: {img_file} ({len(valid_detections)} box(es), score: {max_score:.2f})")

            except Exception as e:
                print(f"  [ERROR] Failed on {img_file}: {e}")
                manual_review.append({"image": img_path, "reason": str(e), "max_score": 0.0})

    # ── Save manual review list ──
    review_path = os.path.join(BASE_DIR, "manual_review.json")
    with open(review_path, "w") as f:
        json.dump(manual_review, f, indent=2)

    # ── Summary ──
    print("\n" + "="*60)
    print(f"✅ Auto-labeled : {total_labeled} images")
    print(f"⏭️  Skipped      : {total_skipped} images (already had XML)")
    print(f"⚠️  Manual review: {len(manual_review)} images → saved to manual_review.json")
    print("="*60)
    print(f"\nOpen manual_review.json to see which images need your attention.")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Point this to your exported saved_model folder
    # After training notebook 2, your model will be here:
    SAVED_MODEL_PATH = os.path.join(
        TENSORFLOW_DIR, "workspace", "models",
        "my_ssd_mobnet", "export", "saved_model"
    )

    if not os.path.exists(SAVED_MODEL_PATH):
        print("[ERROR] Model not found!")
        print(f"Expected at: {SAVED_MODEL_PATH}")
        print("Please train your model first using notebook 2, then run this script.")
    else:
        auto_label(SAVED_MODEL_PATH)