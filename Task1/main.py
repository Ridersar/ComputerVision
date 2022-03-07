import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

x = y = 1600
delta = 64

ev = [-4.0, -3.7, -3.3, -3.0, -2.7, -2.3,
      -2.0, -1.7, -1.3, -1.0, -0.7, -0.3,
      0.0, 0.3, 0.7, 1.0, 1.3, 1.7,
      2.0, 2.3, 2.7, 3.0, 3.3, 3.7, 4.0]
brightness = []

for number in range(1, 26):
    image = cv2.imread("resources/{}.jpg".format(number))
    cropped = image[x:x+delta, y:y+delta]
    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    average_brightness = np.mean(np.mean(gray_cropped))
    average_brightness_log = math.log(average_brightness)
    brightness.append(average_brightness_log)


plt.title("График зависимости яркости от экспозиции")
plt.xlabel("EV")
plt.ylabel("Brightness")
plt.grid()
plt.plot(ev, brightness)
plt.savefig('plot.png')
plt.show()
