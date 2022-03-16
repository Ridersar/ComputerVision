import cv2
import matplotlib.pyplot as plt

image_original = cv2.imread('resources/Forest.jpg')

# 1. Выравнивание гистограммы
image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
cv2.imwrite("1/Result_1.jpg", image_gray)

plt.hist(image_gray.ravel(), 256, [0, 256])
plt.title("Гистограмма до выравнивания")
plt.savefig('1/hist1.png')
plt.close()

plt.hist(image_gray.ravel(), 256, [0, 256], cumulative=True)
plt.title("Гистограмма до выравнивания, кумулятивная")
plt.savefig('1/hist1_cumulative.png')
plt.close()

equalize_image = cv2.equalizeHist(image_gray)
cv2.imwrite("1/Result_2.jpg", equalize_image)

plt.hist(equalize_image.ravel(), 256, [0, 256])
plt.title("Гистограмма после выравнивания")
plt.savefig('1/hist2.png')
plt.close()

plt.hist(equalize_image.ravel(), 256, [0, 256], cumulative=True)
plt.title("Гистограмма после выравнивания, кумулятивная")
plt.savefig('1/hist2_cumulative.png')
plt.close()


# 2. Локальное выравнивание гистограммы над L каналом изображения в LAB (и обратное его преобразование в RGB)
lab_image = cv2.cvtColor(image_original, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab_image)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
equalize_l = clahe.apply(l)
image_lab_res = cv2.merge((equalize_l, a, b))
image_res = cv2.cvtColor(image_lab_res, cv2.COLOR_LAB2BGR)
cv2.imwrite("2/Result.jpg", image_res)


# 3. Гауссовское размытие
image_gauss_blur = cv2.blur(image_original, (10, 10))
cv2.imwrite("3/Gauss.jpg", image_gauss_blur)


# 4. Фильтры Собеля и Лапласа
image_sobel_vertical = cv2.Sobel(image_original, -1, 1, 0)
cv2.imwrite("4/Sobel_vertical.jpg", image_sobel_vertical)
image_sobel_horizontal = cv2.Sobel(image_original, -1, 0, 1)
cv2.imwrite("4/Sobel_horizontal.jpg", image_sobel_horizontal)
image_sobel = cv2.Sobel(image_original, -1, 1, 1)
cv2.imwrite("4/Sobel.jpg", image_sobel)

image_laplace = cv2.Laplacian(image_original, -1)
cv2.imwrite("4/Laplace.jpg", image_laplace)
