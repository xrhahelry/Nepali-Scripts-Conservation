import os
import macros
import cv2
import numpy as np

directory = "Brahmi"
files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

i = 1
k = 1
# Load and preprocess image
for file in files:
    img = cv2.imread(f"{directory}/{file}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter bounding boxes
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 350 < w < 550 and 450 < h < 850:
            boxes.append((x, y, w, h))

    # Sort boxes column
    def sort_boxes(boxes):
        l = int(len(boxes)/6)
        boxes = sorted(boxes, key=lambda b: (b[0] // 20, b[1]))
        boxes = [boxes[i+1:i + 6] for i in range(0, len(boxes), 6)]
        return boxes, l

    sorted_boxes, num_letters = sort_boxes(boxes)

    # Create output directory if not exists
    output_dir = "brahmi_cropped"
    os.makedirs(output_dir, exist_ok=True)
    dir_names = []
    for j in range(0, num_letters):
        dir_names.append(f"char_{k}")
        os.makedirs(f"brahmi_cropped/{dir_names[j]}", exist_ok=True)
        k += 1

    padding = 15
    # Crop and save each box
    folder_num = 0
    for idx, col in enumerate(sorted_boxes):
        for jdf, (x, y, w ,h) in enumerate(col):
            cropped = img[y+padding:y+h-padding, x+padding:x+w-padding]
            crop_dir = output_dir +"/"+ dir_names[folder_num]
            filename = os.path.join(crop_dir, f"crop_{i:02d}.png")
            cv2.imwrite(filename, cropped)
            i += 1
        folder_num += 1
