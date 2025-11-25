import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# ---------- Загрузка изображения ----------
img = cv.imread('Lab2/open-cv-logo.jpg')
assert img is not None, "Файл не найден."

# ---------- 1. Фильтрация с помощью ядра (filter2D) ----------
# Создание усредняющего ядра 5x5
kernel = np.ones((5, 5), np.float32) / 25

# Применение фильтра
dst = cv.filter2D(img, -1, kernel)

# Отображение результата
plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst), plt.title('Averaging (filter2D)')
plt.xticks([]), plt.yticks([])
plt.show()

# ---------- 2. Простое усреднение (cv.blur) ----------
img = cv.imread('Lab2/open-cv-logo.jpg')
assert img is not None, "Файл не найден"
blur = cv.blur(img, (5, 5))

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur), plt.title('Averaging (cv.blur)')
plt.xticks([]), plt.yticks([])
plt.show()

# ---------- 3. Гауссово сглаживание ----------
blur = cv.GaussianBlur(img, (5, 5), 0)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur), plt.title('Gaussian Blur')
plt.xticks([]), plt.yticks([])
plt.show()

# ---------- 4. Медианная фильтрация ----------
# Копия изображения для добавления шума
output = np.copy(img)

# Добавление шума (соль и перец)
prob = 0.5  # вероятность шума 
black = 0
white = 255

# Маски для "соли" (белые точки) и "перца" (черные точки)
salt_mask = np.random.rand(*img.shape[:2]) < (prob / 2)
pepper_mask = np.random.rand(*img.shape[:2]) < (prob / 2)

# Применение шума
output[salt_mask] = white
output[pepper_mask] = black

# Применение медианного фильтра
median = cv.medianBlur(output, 5)

plt.subplot(121), plt.imshow(output), plt.title('Original + Noise')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(median), plt.title('Median Blurring')
plt.xticks([]), plt.yticks([])
plt.show()

# ---------- 5. Двусторонняя фильтрация ----------
# d=9 — диаметр области фильтрации
# sigmaColor=75, sigmaSpace=75 — степень размытия по цвету и пространству
img2 = cv.imread('Lab4/wood.jpg')
blur = cv.bilateralFilter(img2, 9, 75, 75)

plt.subplot(121), plt.imshow(img2), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur), plt.title('Bilateral Filter')
plt.xticks([]), plt.yticks([])
plt.show()
