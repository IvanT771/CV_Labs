import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# ===============================================================
# === 1. Template Matching (поиск одного объекта) ===
# ===============================================================

# --- Загрузка изображения и шаблона ---
img = cv.imread('messi5.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

img2 = img.copy()  # создаём копию, чтобы не изменять исходник

template = cv.imread('template.jpg', cv.IMREAD_GRAYSCALE)
assert template is not None, "template file could not be read, check with os.path.exists()"

w, h = template.shape[::-1]  # ширина и высота шаблона

# --- Список доступных методов сопоставления шаблонов ---
methods = [
    'cv.TM_CCOEFF',
    'cv.TM_CCOEFF_NORMED',
    'cv.TM_CCORR',
    'cv.TM_CCORR_NORMED',
    'cv.TM_SQDIFF',
    'cv.TM_SQDIFF_NORMED'
]

# --- Перебор всех методов и отображение результата ---
for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # cv.matchTemplate: вычисляет степень совпадения между шаблоном и изображением
    res = cv.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    # Для некоторых методов минимум — лучшее совпадение (SQDIFF),
    # для других максимум — лучшее совпадение (CCORR, CCOEFF)
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Рисуем прямоугольник вокруг найденного совпадения
    cv.rectangle(img, top_left, bottom_right, 255, 2)

    # --- Отображаем результат ---
    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

    plt.suptitle(meth)
    plt.show()


# ===============================================================
# === 2. Template Matching with Multiple Objects (поиск нескольких совпадений) ===
# ===============================================================

# --- Загрузка цветного изображения и шаблона ---
img_rgb = cv.imread('mario.png')
assert img_rgb is not None, "file could not be read, check with os.path.exists()"

img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

template = cv.imread('mario_coin.png', 0)
assert template is not None, "template file could not be read, check with os.path.exists()"

w, h = template.shape[::-1]

# --- Выполняем сопоставление с использованием TM_CCOEFF_NORMED ---
res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)

# --- Устанавливаем порог совпадения ---
threshold = 0.8
loc = np.where(res >= threshold)

# --- Рисуем прямоугольники вокруг всех найденных совпадений ---
for pt in zip(*loc[::-1]):  # Переворачиваем координаты (x, y)
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

# --- Отображаем результат ---
plt.imshow(cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB))
plt.title('Detected')
plt.xticks([]), plt.yticks([])
plt.show()
