import numpy as np
import cv2 as cv

# ==== 1) Моменты и центроид ====
img_gray = cv.imread('Lab5/star_c.jpg', cv.IMREAD_GRAYSCALE)
assert img_gray is not None, "file could not be read, check with os.path.exists()"

# Для рисования — цветная версия
img_color = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)

# Бинаризация
ret, thresh = cv.threshold(img_gray, 127, 255, 0)

# Поиск контуров (OpenCV 4.x возвращает 2 значения)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
assert len(contours) > 0, "no contours found"
cnt = contours[0]

M = cv.moments(cnt)
print(M)
cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0

# ==== 2-3) Площадь и периметр ====
area = cv.contourArea(cnt)
perimeter = cv.arcLength(cnt, True)

# ==== 4) Аппроксимация контура (алгоритм Дуглас–Пьюкер) ====
epsilon = 0.1 * cv.arcLength(cnt, True)
approx = cv.approxPolyDP(cnt, epsilon, True)

img_approx = img_color.copy()
cv.drawContours(img_approx, [cnt], -1, (0, 255, 0), 3) 

cv.imshow('Contour Approximation', img_approx)

# ==== 5-6) Выпуклая оболочка и проверка выпуклости ====
hull = cv.convexHull(cnt)
is_convex = cv.isContourConvex(cnt)

# ==== 7a) Ось-ориентированный прямоугольник ====
x, y, w, h = cv.boundingRect(cnt)
cv.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

# ==== 7b) Вращаемый прямоугольник минимальной площади ====
rect = cv.minAreaRect(cnt)              # (center, (w,h), angle)
box = cv.boxPoints(rect)                # 4 угла
box = box.astype(int)
cv.drawContours(img_color, [box], 0, (0, 0, 255), 2)

# ==== 8) Минимальная описанная окружность ====
(xc, yc), radius = cv.minEnclosingCircle(cnt)
center = (int(xc), int(yc))
radius = int(radius)
cv.circle(img_color, center, radius, (255, 0, 0), 2)

# ==== 9) Аппроксимация эллипсом ====
if len(cnt) >= 5:                       # fitEllipse требует >= 5 точек
    ellipse = cv.fitEllipse(cnt)
    cv.ellipse(img_color, ellipse, (0, 255, 255), 2)

# ==== 10) Подбор линии ====
rows, cols = img_gray.shape[:2]
[vx, vy, x0, y0] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)
lefty = int((-x0 * vy / vx) + y0)
righty = int(((cols - x0) * vy / vx) + y0)
cv.line(img_color, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)

# ==== Показ результатов ====
cv.imshow('Features', img_color)
cv.waitKey(0)
cv.destroyAllWindows()
