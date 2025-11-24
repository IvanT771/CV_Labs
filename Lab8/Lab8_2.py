import cv2 as cv
from matplotlib import pyplot as plt

#https://docs.opencv.org/3.4/d5/daf/tutorial_py_histogram_equalization.html

# 1. Загружаем изображение
img = cv.imread('Lab8/t.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "Файл Lab4/noisy2.png не найден."

#cv.imshow("Исходное изображение", img)

# 2. Глобальное выравнивание гистограммы
equ = cv.equalizeHist(img)
#cv.imshow("EqualizeHist", equ)

# 3. CLAHE — адаптивное выравнивание гистограммы
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img)
#cv.imshow("CLAHE", cl1)

# 4. Отобразим гистограммы до и после выравнивания
plt.figure("Глобальное выравнивание гистограммы")
plt.subplot(2, 2, 1)
plt.title("Исходное")
plt.imshow(img, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2)
plt.title("EqualizeHist")
plt.imshow(equ, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3)
plt.title("Hist (original)")
plt.hist(img.ravel(), 256, [0, 256])

plt.subplot(2, 2, 4)
plt.title("Hist (equalized)")
plt.hist(equ.ravel(), 256, [0, 256])

plt.tight_layout()
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()

