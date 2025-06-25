# import cv2
import macros
import cv2
import numpy as np

img = cv2.imread("Brahmi/Copy of IMG_5880.JPG")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boxes = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if 350 < w < 550 and 450 < h < 850:
        print(f"x={x}, y={y}, w={w}, h={h}")
        boxes.append((x, y, w, h))

def sort_boxes(boxes, row_tolerance=20):
    boxes = sorted(boxes, key=lambda b: (b[1] // row_tolerance, b[0]))
    return boxes

sorted_boxes = sort_boxes(boxes)

preview = img.copy()
for (x, y, w, h) in sorted_boxes:
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)

macros.display_image(preview)
