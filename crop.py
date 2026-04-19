"""
crops faces from all images in class_pictures/ using opencv haar cascade.
saves cropped images in place (overwrites originals).
images where no face is detected are skipped and reported at the end.
"""

import cv2
import os
from pathlib import Path

DATASET_PATH = Path("class_pictures")
CASCADE_PATH  = "haarcascade_frontalface_default.xml"
PADDING       = 0.3   # fraction of face size to pad on each side
OUTPUT_SIZE   = (128, 128)

detector = cv2.CascadeClassifier(CASCADE_PATH)
failed   = []

for img_path in sorted(DATASET_PATH.glob("*/*.jpg")):
    img   = cv2.imread(str(img_path))
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        print(f"  [no face] {img_path}")
        failed.append(img_path)
        continue

    # use the largest detected face if multiple found
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # apply padding
    pad_x = int(w * PADDING)
    pad_y = int(h * PADDING)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img.shape[1], x + w + pad_x)
    y2 = min(img.shape[0], y + h + pad_y)

    cropped = img[y1:y2, x1:x2]
    resized = cv2.resize(cropped, OUTPUT_SIZE)
    cv2.imwrite(str(img_path), resized)
    print(f"  [ok] {img_path}")

print(f"\ndone. {len(failed)} image(s) skipped:")
for p in failed:
    print(f"  {p}")
