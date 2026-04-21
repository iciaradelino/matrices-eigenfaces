import os
import sys
from pathlib import Path

import cv2
import numpy as np

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
OUTPUT_SIZE = 192
MARGIN = 0.30

detector = cv2.CascadeClassifier(CASCADE_PATH)
if detector.empty():
    raise RuntimeError(f"Could not load cascade file: {CASCADE_PATH}")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def center_square_crop(img, frac=0.88):
    h, w = img.shape[:2]
    side = int(min(h, w) * frac)
    cx, cy = w // 2, h // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    return img[y1:y2, x1:x2]


def square_crop_from_face(img, x, y, w, h, margin=MARGIN):
    H, W = img.shape[:2]
    cx = x + w / 2
    cy = y + h / 2
    side = int(max(w, h) * (1 + 2 * margin))

    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x2 = x1 + side
    y2 = y1 + side

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > W:
        shift = x2 - W
        x1 -= shift
        x2 = W
    if y2 > H:
        shift = y2 - H
        y1 -= shift
        y2 = H

    x1 = max(0, x1)
    y1 = max(0, y1)

    crop = img[y1:y2, x1:x2]
    h2, w2 = crop.shape[:2]
    side2 = min(h2, w2)
    crop = crop[:side2, :side2]
    return crop


def detect_face(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    faces = detector.detectMultiScale(
        gray_eq,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(40, 40)
        )

    if len(faces) == 0:
        return None

    return max(faces, key=lambda b: b[2] * b[3])


def process_image(in_path, out_path):
    img = cv2.imread(str(in_path))
    if img is None:
        print(f"[bad file] {in_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = detect_face(gray)

    if face is None:
        crop = center_square_crop(gray)
        tag = "fallback"
    else:
        x, y, w, h = face
        crop = square_crop_from_face(gray, x, y, w, h)
        tag = "ok"

    interp = cv2.INTER_CUBIC if crop.shape[0] < OUTPUT_SIZE else cv2.INTER_AREA
    out = cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=interp)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(str(out_path), out)
    print(f"[{tag}] {in_path}")


def main():
    if len(sys.argv) < 2:
        print("usage: python crop.py <input_folder> [output_folder]")
        sys.exit(1)

    in_root = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        out_root = Path(sys.argv[2])
    else:
        out_root = Path(str(in_root) + "_cropped")

    for root, _, files in os.walk(in_root):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            in_path = Path(root) / fname
            rel = in_path.relative_to(in_root)
            out_path = out_root / rel
            process_image(in_path, out_path)


if __name__ == "__main__":
    main()