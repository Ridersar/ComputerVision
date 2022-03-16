# С помощью лапласовской пирамиды склеить 2 изображения по маске
import cv2
import numpy as np

A = cv2.imread('resources/tomato.jpg')
B = cv2.imread('resources/lemon.jpg')

_, cols, _ = A.shape
original_blending = np.hstack((A[:, :cols//2], B[:, cols//2:]))

# Пирамида Гаусса (A)
gpA = [A]
for i in range(8):
    A = cv2.pyrDown(A)
    gpA.append(A)

# Пирамида Гаусса (B)
gpB = [B]
for i in range(8):
    B = cv2.pyrDown(B)
    gpB.append(B)

# Пирамида Лапласа (A)
lpA = [gpA[7]]
for i in range(7, 0, -1):
    gaussian_expanded = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i-1], gaussian_expanded)
    lpA.append(L)

# Пирамида Лапласа (B)
lpB = [gpB[7]]
for i in range(7, 0, -1):
    gaussian_expanded = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i-1], gaussian_expanded)
    lpB.append(L)

# Склеиваем половины изображений на каждом уровне
lp = []
for la, lb in zip(lpA, lpB):
    _, cols, _ = la.shape
    ls = np.hstack((la[:, :cols//2], lb[:, cols//2:]))
    lp.append(ls)

# Воссоздание изображения
result_blending = lp[0]
for i in range(1, 8):
    result_blending = cv2.pyrUp(result_blending)
    result_blending = cv2.add(result_blending, lp[i])

cv2.imwrite('Result_blending.jpg', result_blending)
cv2.imwrite('Original_blending.jpg', original_blending)

