import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# ================================================================
#  ПЕРВЫЙ ПРИМЕР — ЛАПЛАС И СОБЕЛЬ (из dave.jpg)
# ================================================================

# Загружаем изображение в оттенках серого
img = cv.imread('Lab3/sudoku.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# --- Laplacian ---
# Вычисляет вторую производную (чувствителен к резким перепадам яркости)
laplacian = cv.Laplacian(img, cv.CV_64F)

# --- Sobel X и Sobel Y ---
# Первая производная по X и по Y соответственно
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

# --- Визуализация ---
plt.figure(figsize=(10,8))
plt.subplot(2,2,1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,2), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,3), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,4), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()


# ================================================================
#  ВТОРОЙ ПРИМЕР — РАЗНИЦА МЕЖДУ CV_8U и CV_64F (из box.png)
# ================================================================

# Загружаем второе изображение
img2 = cv.imread('Lab6/figure.jpg', cv.IMREAD_GRAYSCALE)
assert img2 is not None, "file could not be read, check with os.path.exists()"

# --- Sobel с типом CV_8U ---
# Отрицательные значения усекаются (обрезаются до 0)
sobelx8u = cv.Sobel(img2, cv.CV_8U, 1, 0, ksize=5)

# --- Sobel с типом CV_64F ---
# Сохраняет отрицательные значения, затем берём модуль и преобразуем в uint8
sobelx64f = cv.Sobel(img2, cv.CV_64F, 1, 0, ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)

# --- Визуализация ---
plt.figure(figsize=(10,4))
plt.subplot(1,3,1), plt.imshow(img2, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(1,3,2), plt.imshow(sobelx8u, cmap='gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])

plt.subplot(1,3,3), plt.imshow(sobel_8u, cmap='gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
