import cv2 as cv
import numpy as np

# === ЛР9_1. Детектор углов Харриса
# === По примеру из:
# === https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html

# 1. Загружаем изображение с большим количеством пересечений (углов)
img = cv.imread('Lab9/l9.jpg')
assert img is not None, "Файл Lab3/sudoku.jpg не найден."

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. Переводим изображение в формат float32, как требует cornerHarris
gray = np.float32(gray)

# 3. Вычисляем отклик детектора Харриса
#    blockSize=2, ksize=3, k=0.04 — как в примере из документации
dst = cv.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# 4. Увеличиваем отклик для лучшей визуализации
dst = cv.dilate(dst, None)

# 5. Отмечаем найденные углы красным цветом
img[dst > 0.01 * dst.max()] = [0, 0, 255]

cv.imshow('Corners (Harris)', img)
cv.waitKey(0)
cv.destroyAllWindows()

