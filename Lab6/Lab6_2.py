import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# === Загрузка изображения в оттенках серого ===
img = cv.imread('Lab6/car.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# === 1. Применение размытия ===
# Уменьшает шум перед применением детектора контуров
img = cv.GaussianBlur(img, (5,5), 0)

# === 2. Применение оператора Canny ===
# cv.Canny(источник, нижний_порог, верхний_порог)
# Пороговые значения подбираются эмпирически:
#   - Нижний порог: минимальная величина градиента для учета границы
#   - Верхний порог: порог для уверенных границ
edges = cv.Canny(img, 100, 200)

# === 3. Отображение результатов ===
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
