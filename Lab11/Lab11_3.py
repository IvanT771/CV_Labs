import cv2 as cv
from matplotlib import pyplot as plt

# === ЛР11_3. Удаление шума методом нелокальных средних
# === По примеру из:
# === https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html

# 1. Загружаем зашумлённое изображение
img = cv.imread('Lab8/t.jpg')
assert img is not None, "Файл Lab8/t.jpg не найден."

# 2. Преобразуем в пространство HSV и/или работаем сразу в BGR
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 3. Применяем fastNlMeansDenoisingColored
dst = cv.fastNlMeansDenoisingColored(
    img, None,
    h=10, hColor=10,      # параметры фильтра по яркости и цвету
    templateWindowSize=7,
    searchWindowSize=21
)

dst_rgb = cv.cvtColor(dst, cv.COLOR_BGR2RGB)

# 4. Отображаем результат
plt.figure("Non-local Means Denoising")
plt.subplot(1, 2, 1)
plt.title("Исходное (с шумом)")
plt.imshow(img_rgb)
plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2)
plt.title("После fastNlMeansDenoisingColored")
plt.imshow(dst_rgb)
plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

