"""
crops faces from all images in a dataset folder using opencv haar cascade.
handles both flat folders (celebrity_pictures/*.jpg)
and per-person subfolders (class_pictures/person/*.jpg).
saves cropped images in place (overwrites originals).
images where no face is detected are skipped and reported at the end.

usage:
    python crop.py class_pictures
    python crop.py celebrity_pictures
"""

import sys
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

CASCADE_PATH = "haarcascade_frontalface_default.xml"
PADDING      = 0.3   # fraction of face size to pad on each side
OUTPUT_SIZE  = (128, 128)

dataset_path = Path(sys.argv[1] if len(sys.argv) > 1 else "class_pictures")
images       = sorted(dataset_path.glob("**/*.jpg")) + sorted(dataset_path.glob("**/*.jpeg"))
detector     = cv2.CascadeClassifier(CASCADE_PATH)
failed       = []

for img_path in images:
    try:
        img = cv2.cvtColor(np.array(Image.open(img_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"  [error] {img_path}: {e}")
        failed.append(img_path)
        continue

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        print(f"  [no face] {img_path}")
        failed.append(img_path)
        continue

    # use the largest detected face if multiple found
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    pad_x = int(w * PADDING)
    pad_y = int(h * PADDING)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img.shape[1], x + w + pad_x)
    y2 = min(img.shape[0], y + h + pad_y)

    cropped = img[y1:y2, x1:x2]
    resized = cv2.resize(cropped, OUTPUT_SIZE)
    Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)).save(img_path)
    print(f"  [ok] {img_path}")

print(f"\ndone. {len(failed)} image(s) skipped:")
for p in failed:
    print(f"  {p}")
