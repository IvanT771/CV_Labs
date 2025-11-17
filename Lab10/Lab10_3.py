import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# === ЛР10_3. ORB — ориентированный FAST и вращательный BRIEF
# === https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html

# 1. Загружаем изображение
img = cv.imread('Lab9/l9.jpg')
assert img is not None, "Файл Lab3/img2.jpg не найден."

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. Создаём объект ORB
orb = cv.ORB_create()

# 3. Детектируем ключевые точки и вычисляем дескрипторы
keypoints, descriptors = orb.detectAndCompute(gray, None)

print("Найдено ключевых точек:", len(keypoints))
print("Размер матрицы дескрипторов:", descriptors.shape if descriptors is not None else None)

# 4. Отрисовываем ключевые точки
img2 = cv.drawKeypoints(img, keypoints, None, color=(0,255,0), flags=0)
plt.imshow(img2), plt.show()

