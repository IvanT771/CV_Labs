import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# ---------- Загрузка и преобразование изображения ----------
# Считываем изображение в оттенках серого
img = cv.imread('Lab4/figure.jpg')
assert img is not None, "Файл не найден."

imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# ---------- Пороговая обработка ----------
# Преобразуем изображение в бинарное (чёрно-белое)
ret, thresh = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY)

# ---------- Поиск контуров ----------
# cv.findContours возвращает список контуров и иерархию
# Режим RETR_TREE — возвращает все контуры и строит полную иерархию
# CHAIN_APPROX_SIMPLE — сжимает горизонтальные, вертикальные и диагональные сегменты
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# ---------- Рисование контуров ----------
# Копия исходного изображения для рисования
img_contours = cv.cvtColor(imgray, cv.COLOR_GRAY2BGR)
cv.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

# ---------- Отображение ----------
titles = ['Original Image', 'Thresholded', 'Contours']
images = [img, thresh, img_contours]

for i in range(3):
    plt.subplot(1, 3, i + 1)
    if i < 2:
        plt.imshow(images[i], 'gray')
    else:
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
