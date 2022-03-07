import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


image = cv2.imread('resources/Forest.jpg')

hsv_image1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv_image1)
value = 30
s += value

hsv_image1 = cv2.merge((h, s, v))

hsv_image2 = cv2.cvtColor(hsv_image1, cv2.COLOR_HSV2BGR)

cv2.imwrite("Result.jpg", hsv_image2)

cv2.imshow("Image result", hsv_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
