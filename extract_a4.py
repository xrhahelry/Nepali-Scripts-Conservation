import cv2
import macros
import numpy as np

# IMG_20250623_162758_1.jpg
# IMG_20250623_162754.jpg
img = cv2.imread("IMG_20250623_162758_1.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# edges = cv2.Canny(gray_img, 30, 80)

_, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, -1, (0,255,0), 2)
largest_contour = max(contours, key=cv2.contourArea)
mask = np.zeros_like(gray_img)
cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
extracted = cv2.bitwise_and(img, img, mask=mask)
x, y, w, h = cv2.boundingRect(largest_contour)
cropped = extracted[y:y+h, x:x+w]

macros.display_image(cropped)
