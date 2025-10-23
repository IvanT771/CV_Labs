#Lab3_1 Basic Operations on Images

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#1. Загрузка изображения 
img = cv.imread('Lab2/open-cv-logo.jpg') 
if img is None:
    raise FileNotFoundError("Ошибка: изображение не найдено.")

cv.imshow("Original", img)
print("Размер изображения:", img.shape)

#2. Доступ к пикселям и изменение 
px = img[100, 100]
print("Пиксель (100,100):", px)
print("Синий канал:", img[100, 100, 0])

# Меняем пиксель
img[100, 100] = [0, 255, 0]
print("После изменения:", img[100, 100])

#3. Работа с ROI 
roi = img[48:208,176:336]  # выделяем область
cv.imshow("ROI", roi)

#Копируем ROI в другое место
img[200:360,264:424] = roi
cv.imshow("ROI Copied", img)

#4. Разделение и объединение каналов
b, g, r = cv.split(img)
# cv.imshow("Blue Channel", b)
# cv.imshow("Green Channel", g)
cv.imshow("Red Channel", r)

merged = cv.merge([b, g, r])
cv.imshow("Merged Image", merged)

#5. Работа с отдельными каналами
img[:, :, 2] = 0  # обнуляем красный канал
cv.imshow("No Red Channel", img)

BLUE = [255,0,0]
 
img1 = cv.imread('Lab2/open-cv-logo.jpg')
assert img1 is not None, "file could not be read, check with os.path.exists()"
 
replicate = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REPLICATE)
reflect = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT)
reflect101 = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT_101)
wrap = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_WRAP)
constant= cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_CONSTANT,value=BLUE)
 
plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
 
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
