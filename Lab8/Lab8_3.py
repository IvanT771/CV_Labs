import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# === ЛР8_3. Двумерные гистограммы 
# === спользуются для построения цветовых гистограмм, где двумя признаками являются значения оттенка и насыщенности каждого пикселя.
# === https://docs.opencv.org/3.4/dd/d0d/tutorial_py_2d_histogram.html)

# 1. Загружаем цветное изображение
img = cv.imread('Lab1/test.jpg')
assert img is not None, "Файл Lab3/img2.jpg не найден."

cv.imshow("Исходное изображение", img)

# 2. Преобразуем изображение в цветовое пространство HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# 3. Считаем двумерную гистограмму по каналам H и S
hist = cv.calcHist(
    [hsv],        # изображение
    [0, 1],       # каналы (H и S)
    None,         # маска
    [180, 256],   # количество корзин по H и S
    [0, 180, 0, 256]  # диапазоны значений
)

# 4. Отображаем двумерную гистограмму
plt.figure("2D гистограмма H-S")
plt.imshow(hist, interpolation='nearest')
plt.title("2D гистограмма (H-S)")
plt.xlabel("S")
plt.ylabel("H")
plt.colorbar()

plt.show()
cv.waitKey(0)
cv.destroyAllWindows()

