import cv2 as cv
import numpy as np

# === ЛР12_3. Обнаружение окружностей (Хафф)
# === По примеру из:
# === https://docs.opencv.org/3.4/da/d53/tutorial_py_houghcircles.html

# 1. Загружаем изображение с круглыми объектами
img = cv.imread('Lab6/figure.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "Файл Lab6/figure.jpg не найден."

# 2. Небольшое сглаживание для уменьшения шума
img_blur = cv.medianBlur(img, 5)

# 3. Преобразуем в BGR для рисования цветных окружностей
cimg = cv.cvtColor(img_blur, cv.COLOR_GRAY2BGR)

# 4. Поиск окружностей методом Хаффа
circles = cv.HoughCircles(
    img_blur,
    cv.HOUGH_GRADIENT,
    dp=1,
    minDist=20,
    param1=50,
    param2=30,
    minRadius=0,
    maxRadius=0
)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Внешняя окружность
        cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Центр окружности
        cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

cv.imshow('Original (blurred)', img_blur)
cv.imshow('Hough Circles', cimg)
cv.waitKey(0)
cv.destroyAllWindows()

