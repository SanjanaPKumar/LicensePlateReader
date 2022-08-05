# code referenced from https://github.com/nicknochnack/ANPRwithPython

import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

states = ["CALIFORNIA", "VIRGINIA", "NEW YORK", "MICHIGAN", "MASSACHUSETTES"]

image_path = "assets/ca-license-plate-6.jpeg"
img = cv2.imread(image_path)

# convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

# smoothen the image & detect edges
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)
# plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

# find contours in image
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# find rectangles made of contours with aspect ratio between 1.6 and 2.4
aspect_ratio = 0.0
img_copy = img.copy()
max_area = 0

chosen_x = 0
chosen_y = 0
chosen_w = 0
chosen_h = 0

chosen_contour = contours[0]
  for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    if aspect_ratio >= 1.6 and aspect_ratio <= 2.4:
        area = w * h

        # choose rectangle with the largest area
        if max_area < area:
            chosen_contour = contour
            max_area = area
            
            chosen_x = x
            chosen_y = y
            chosen_w = w
            chosen_h = h

cv2.rectangle(
    img_copy,
    (chosen_x, chosen_y),
    (chosen_x + chosen_w, chosen_y + chosen_h),
    (0, 255, 0),
    2,
)

# crop image based on chosen rectangle
cropped_image = img_copy[chosen_y : chosen_y + chosen_h, chosen_x : chosen_x + chosen_w]

# plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

# use easyocr to read the license plate text
reader = easyocr.Reader(["en"])
result = reader.readtext(cropped_image)
# print(result)

# find the final plate text

plate = "invalid"
max_accuracy = 0
for i in result:
    if len(i[-2]) >= 6 and len(i[-2]) <= 8:
        text = i[-2]
        # print(text)
        if text.upper() not in states:
            if max_accuracy < i[-1]:
                max_accuracy = i[-1]
                plate = text

print("License plate found: ", plate)
