import cv2 as cv
import numpy as np

# === ЛР12_2. Обнаружение прямых линий 
# === По примеру из:
# === https://docs.opencv.org/3.4/d6/d10/tutorial_py_houghlines.html

# 1. Загружаем изображение судоку (много прямых линий)
img = cv.imread('Lab3/sudoku.jpg')
assert img is not None, "Файл Lab3/sudoku.jpg не найден."

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. Находим границы (Canny)
edges = cv.Canny(gray, 50, 150, apertureSize=3)

#cv.imshow('Edges', edges)

# 3. Классический Хафф (HoughLines)
lines = cv.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=200)

img_hough = img.copy()
if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(img_hough, (x1, y1), (x2, y2), (0, 0, 255), 2)

#cv.imshow('HoughLines', img_hough)

# 4. Пробабилистический Хафф (HoughLinesP)
lines_p = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

img_hough_p = img.copy()
if lines_p is not None:
    for l in lines_p:
        x1, y1, x2, y2 = l[0]
        cv.line(img_hough_p, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv.imshow('HoughLinesP', img_hough_p)

cv.waitKey(0)
cv.destroyAllWindows()

