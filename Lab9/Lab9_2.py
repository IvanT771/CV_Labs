import cv2 as cv
import numpy as np

# === ЛР9_2. Детектор углов Ши–Томаси
# === https://docs.opencv.org/3.4/d4/d8c/tutorial_py_shi_tomasi.html

# 1. Загружаем изображение
img = cv.imread('Lab9/l9.jpg')
assert img is not None, "Файл Lab3/sudoku.jpg не найден."

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. Поиск хороших углов для отслеживания
corners = cv.goodFeaturesToTrack(
    gray,
    maxCorners=50,     # максимальное число углов
    qualityLevel=0.01, # минимальное качество угла
    minDistance=10     # минимальное расстояние между углами
)

corners = np.rint(corners).astype(np.int32)

# 3. Рисуем найденные углы
for i in corners:
    x, y = i.ravel()
    cv.circle(img, (x, y), 5, (0, 0, 255), -1)

cv.imshow('Shi-Tomasi corners', img)
cv.waitKey(0)
cv.destroyAllWindows()
