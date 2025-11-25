import cv2 as cv

# === ЛР11_1. Сопоставление дескрипторов (Brute-Force Matcher)
# === По примеру из:
# === https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
#
# В примере используется ORB и BFMatcher с нормой Hamming.

# 1. Загружаем две картинки
#    Для демонстрации берём одно и то же изображение, чтобы
#    было много совпадающих ключевых точек.
img1 = cv.imread('Lab3/img2.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('Lab3/img2.jpg', cv.IMREAD_GRAYSCALE)

assert img1 is not None and img2 is not None, "Файлы Lab3/img2.jpg не найдены."

# 2. Создаём извлекатель признаков ORB
orb = cv.ORB_create()

# 3. Находим ключевые точки и дескрипторы
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

print("Ключевых точек в img1:", len(kp1))
print("Ключевых точек в img2:", len(kp2))

# 4. Создаём Brute-Force matcher с нормой Hamming
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# 5. Выполняем сопоставление дескрипторов
matches = bf.match(des1, des2)

# Сортируем по расстоянию (чем меньше, тем лучше)
matches = sorted(matches, key=lambda x: x.distance)

# 6. Рисуем первые 20 совпадений
img_matches = cv.drawMatches(
    img1, kp1,
    img2, kp2,
    matches[:20],
    None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

cv.imshow("ORB + BFMatcher (первые 20 совпадений)", img_matches)
cv.waitKey(0)
cv.destroyAllWindows()

