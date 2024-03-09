import os
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve

matplotlib.rcParams['figure.figsize'] = (9.0, 9.0)

image = cv2.imread("Apollo_11_Launch.jpg", cv2.IMREAD_COLOR)

# Display the original image
plt.imshow(image[:, :, ::-1])

# Draw a line
imageLine = image.copy()
cv2.line(imageLine, (200, 100), (400, 100), color=(0, 255, 255), thickness=5, lineType=cv2.LINE_AA)

# Display the image
plt.imshow(imageLine[:,:,::-1])

# Draw a circle
imageCircle = image.copy()

cv2.circle(imageCircle, (900,500), 100, color=(0, 0, 255), thickness=5, lineType=cv2.LINE_AA)

# Display the image
plt.imshow(imageCircle[:,:,::-1])

# Rectangle
imageRectangle = image.copy()

cv2.rectangle(imageRectangle, (500, 100), (700, 600), (255, 0, 255), thickness=5, lineType=cv2.LINE_8)

# Display the image
plt.imshow(imageRectangle[:, :, ::-1])

# Adding Text
imageText = image.copy()
text = "Apollo 11 Saturn V Launch, July 16, 1969"
fontScale = 2.3
fontFace = cv2.FONT_HERSHEY_PLAIN
fontColor = (0, 255, 0)
fontThickness = 2

cv2.putText(imageText, text, (200, 700), fontFace, fontScale, fontColor, fontThickness, cv2.LINE_AA);

# Display the image
plt.imshow(imageText[:, :, ::-1])