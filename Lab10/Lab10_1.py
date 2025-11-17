import cv2 as cv

# === ЛР10_1. FAST — быстрый детектор ключевых точек
# === https://docs.opencv.org/3.4/df/d0c/tutorial_py_fast.html

# 1. Загружаем изображение в оттенках серого
img = cv.imread('Lab9/l9.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "Файл Lab3/img2.jpg не найден."

# 2. Создаём детектор FAST
fast = cv.FastFeatureDetector_create()

# 3. Детектируем ключевые точки
kp = fast.detect(img, None)

print("Всего ключевых точек (с nonmaxSuppression):", len(kp))

# 4. Рисуем ключевые точки
img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
cv.imshow('FAST keypoints (nonmaxSuppression = True)', img2)

# 5. Сравнение с отключённым nonmaxSuppression
fast.setNonmaxSuppression(False)
kp2 = fast.detect(img, None)
print("Ключевых точек (nonmaxSuppression = False):", len(kp2))

img3 = cv.drawKeypoints(img, kp2, None, color=(0, 255, 0))
cv.imshow('FAST keypoints (nonmaxSuppression = False)', img3)

cv.waitKey(0)
cv.destroyAllWindows()

