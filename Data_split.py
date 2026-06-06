"""
Data Split Script
-----------------
Split 1: train_label / test_label  → for initial training (powers auto-labeling)
Split 2: train / test              → for final object detection training
"""

import os
import shutil
import random

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE = r"E:\Study\Projects\REAL-TIME OBJECT DETECTION FOR ASSISTING COMMUNICATION IN NON-VERBAL INDIVIDUALS\Tensorflow\workspace\images"

COLLECTED_DIR   = os.path.join(BASE, "collectedimages")

# Split 1 — initial training (manually labeled only)
TRAIN_LABEL_DIR = os.path.join(BASE, "train_label")
TEST_LABEL_DIR  = os.path.join(BASE, "test_label")

# Split 2 — final training (all labeled images)
TRAIN_DIR       = os.path.join(BASE, "train")
TEST_DIR        = os.path.join(BASE, "test")

CLASSES         = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SPLIT_RATIO     = 0.8   # 80% train, 20% test

# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def copy_pairs(images, src_dir, dest_dir):
    """Copy image + xml pairs to destination folder."""
    for img in images:
        xml = img.replace(".jpg", ".xml")
        shutil.copy(os.path.join(src_dir, img), dest_dir)
        xml_src = os.path.join(src_dir, xml)
        if os.path.exists(xml_src):
            shutil.copy(xml_src, dest_dir)

def split_and_copy(images, src_dir, train_dir, test_dir, class_name, split_type):
    random.shuffle(images)
    split_idx    = int(SPLIT_RATIO * len(images))
    train_images = images[:split_idx]
    test_images  = images[split_idx:]

    copy_pairs(train_images, src_dir, train_dir)
    copy_pairs(test_images,  src_dir, test_dir)

    print(f"  [{split_type}] {class_name}: {len(train_images)} train, {len(test_images)} test")
    return len(train_images), len(test_images)

# ─────────────────────────────────────────────
# CREATE DIRECTORIES
# ─────────────────────────────────────────────
for d in [TRAIN_LABEL_DIR, TEST_LABEL_DIR, TRAIN_DIR, TEST_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────
# MAIN SPLIT
# ─────────────────────────────────────────────
total_label_train = 0
total_label_test  = 0
total_train       = 0
total_test        = 0

print("=" * 55)
print("SPLIT 1 — Initial Training (manually labeled only)")
print("=" * 55)

for class_name in CLASSES:
    class_dir = os.path.join(COLLECTED_DIR, class_name)
    if not os.path.isdir(class_dir):
        print(f"  [SKIP] {class_name} — folder not found")
        continue

    # Get ONLY images that have XML (manually labeled)
    all_images = [f for f in os.listdir(class_dir) if f.endswith(".jpg")]
    labeled    = [f for f in all_images
                  if os.path.exists(os.path.join(class_dir, f.replace(".jpg", ".xml")))]
    unlabeled  = [f for f in all_images if f not in labeled]

    if len(labeled) == 0:
        print(f"  [SKIP] {class_name} — no labeled images found")
        continue

    tr, te = split_and_copy(labeled, class_dir,
                             TRAIN_LABEL_DIR, TEST_LABEL_DIR,
                             class_name, "label")
    total_label_train += tr
    total_label_test  += te

print(f"\n✅ Split 1 Done → train_label: {total_label_train}, test_label: {total_label_test}")

print("\n" + "=" * 55)
print("SPLIT 2 — Final Training (ALL labeled images)")
print("=" * 55)
print("⚠️  Run this AFTER auto-labeling is complete!\n")

confirm = input("Have you completed auto-labeling? (yes/no): ").strip().lower()

if confirm == "yes":
    for class_name in CLASSES:
        class_dir = os.path.join(COLLECTED_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Get ALL images that have XML (manual + auto-labeled)
        all_labeled = [f for f in os.listdir(class_dir)
                       if f.endswith(".jpg") and
                       os.path.exists(os.path.join(class_dir, f.replace(".jpg", ".xml")))]

        if len(all_labeled) == 0:
            print(f"  [SKIP] {class_name} — no labeled images found")
            continue

        tr, te = split_and_copy(all_labeled, class_dir,
                                 TRAIN_DIR, TEST_DIR,
                                 class_name, "final")
        total_train += tr
        total_test  += te

    print(f"\n✅ Split 2 Done → train: {total_train}, test: {total_test}")
else:
    print("\n⏭️  Skipping Split 2 — run this script again after auto-labeling.")

print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"train_label/ : {total_label_train} images  ← use for initial training")
print(f"test_label/  : {total_label_test} images   ← use for testing initial model")
print(f"train/       : {total_train} images         ← use for final training")
print(f"test/        : {total_test} images          ← use for final testing")
print("=" * 55)