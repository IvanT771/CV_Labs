import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#https://docs.opencv.org/3.4/d1/db7/tutorial_py_histogram_begins.html
#общее представление о распределении интенсивности изображения

# 1. Загружаем изображение в оттенках серого
img = cv.imread('Lab3/img2.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "Файл Lab3/img2.jpg не найден."

cv.imshow("Исходное изображение", img)

# 2. Построение простой гистограммы яркости с помощью matplotlib
plt.figure("Простая гистограмма (plt.hist)")
plt.title("Гистограмма яркости (plt.hist)")
plt.xlabel("Значение пикселя")
plt.ylabel("Количество пикселей")
plt.hist(img.ravel(), 256, [0, 256])
plt.xlim([0, 256])

# 3. Построение гистограммы с помощью cv.calcHist
hist = cv.calcHist([img], [0], None, [256], [0, 256])

plt.figure("Гистограмма (cv.calcHist)")
plt.title("Гистограмма яркости (cv.calcHist)")
plt.xlabel("Значение пикселя")
plt.ylabel("Количество пикселей")
plt.plot(hist)
plt.xlim([0, 256])

# 4. Построим маску и гистограмму только по области интереса
mask = np.zeros(img.shape[:2], np.uint8)
rows, cols = img.shape

# Выделим прямоугольную область в центре изображения
center_row_start = rows // 4
center_row_end = 3 * rows // 4
center_col_start = cols // 4
center_col_end = 3 * cols // 4
mask[center_row_start:center_row_end, center_col_start:center_col_end] = 255

masked_img = cv.bitwise_and(img, img, mask=mask)

#cv.imshow("Маска", mask)
#cv.imshow("Изображение с маской", masked_img)

hist_mask = cv.calcHist([img], [0], mask, [256], [0, 256])

plt.figure("Сравнение гистограмм")
plt.title("Гистограмма всего изображения и по маске")
plt.plot(hist, label="Полное изображение")
plt.plot(hist_mask, label="По маске")
plt.xlim([0, 256])
plt.legend()

plt.show()
cv.waitKey(0)
cv.destroyAllWindows()

