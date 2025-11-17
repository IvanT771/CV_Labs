import cv2 as cv

# === ЛР9_3. SIFT — масштабно-инвариантные признаки
# === https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html

# 1. Загружаем изображение
img = cv.imread('Lab3/img2.jpg')
assert img is not None, "Файл Lab3/img2.jpg не найден."

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. Создаём объект SIFT
sift = cv.SIFT_create()

# 3. Детектируем ключевые точки и вычисляем дескрипторы
keypoints, descriptors = sift.detectAndCompute(gray, None)

print(f"Найдено ключевых точек: {len(keypoints)}")

# 4. Рисуем ключевые точки на изображении
img_keypoints = cv.drawKeypoints(
    img, keypoints, None,
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv.imshow('SIFT keypoints', img_keypoints)
cv.waitKey(0)
cv.destroyAllWindows()

