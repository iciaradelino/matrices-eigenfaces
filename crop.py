"""
crop faces from all images in a dataset folder using opencv haar cascade.
handles both flat folders (celebrity_pictures/*.jpg)
and per-person subfolders (class_pictures/person/*.jpg).
saves cropped images in place (overwrites originals).

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
OUTPUT_SIZE = (128, 128)
PADDING = 0.22

dataset_path = Path(sys.argv[1] if len(sys.argv) > 1 else "class_pictures")
images = sorted(dataset_path.glob("**/*.jpg")) + sorted(dataset_path.glob("**/*.jpeg")) + sorted(dataset_path.glob("**/*.png"))
detector = cv2.CascadeClassifier(CASCADE_PATH)
failed = []

def detect_face(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)

    faces = detector.detectMultiScale(
        eq, scaleFactor=1.08, minNeighbors=6, minSize=(60, 60)
    )
    if len(faces) == 0:
        faces = detector.detectMultiScale(
            eq, scaleFactor=1.05, minNeighbors=4, minSize=(50, 50)
        )
    return faces

for img_path in images:
    try:
        img = cv2.cvtColor(np.array(Image.open(img_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[error] {img_path}: {e}")
        failed.append(img_path)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_face(gray)

    if len(faces) == 0:
        print(f"[no face] {img_path}")
        failed.append(img_path)
        continue

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    pad_x = int(w * PADDING)
    pad_y = int(h * PADDING)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img.shape[1], x + w + pad_x)
    y2 = min(img.shape[0], y + h + pad_y)

    # force square crop
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    side = max(x2 - x1, y2 - y1)

    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(img.shape[1], x1 + side)
    y2 = min(img.shape[0], y1 + side)

    # re-adjust if clipped by border
    side = min(x2 - x1, y2 - y1)
    x2 = x1 + side
    y2 = y1 + side

    cropped = img[y1:y2, x1:x2]
    resized = cv2.resize(cropped, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
    Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)).save(img_path)
    print(f"[ok] {img_path}")

print(f"\ndone. {len(failed)} image(s) skipped:")
for p in failed:
    print(f"  {p}")