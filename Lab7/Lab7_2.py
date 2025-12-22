import cv2 as cv
import numpy as np

#https://docs.opencv.org/4.x/dd/dd7/tutorial_morph_lines_detection.html
#Извлечение горизонтальных и вертикальных линий с помощью морфологических операций

# === Загрузка изображения ===
src = cv.imread('Lab3/sudoku.jpg', cv.IMREAD_GRAYSCALE)
assert src is not None, "Файл не найден."

# === 1. Инвертируем изображение и применяем адаптивный порог ===
src_bin = cv.adaptiveThreshold(~src, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)

# === 2. Извлечение горизонтальных и вертикальных линий ===
horizontal = src_bin.copy()
vertical = src_bin.copy()

# Размер структурирующего элемента
cols = horizontal.shape[1]
horizontal_size = cols // 30

horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
horizontal = cv.erode(horizontal, horizontalStructure)
horizontal = cv.dilate(horizontal, horizontalStructure)

rows = vertical.shape[0]
verticalsize = rows // 30

verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
vertical = cv.erode(vertical, verticalStructure)
vertical = cv.dilate(vertical, verticalStructure)

# === 3. Комбинируем линии ===
mask = horizontal + vertical

# === 4. Отображаем результаты ===
#cv.imshow("Original", src)
#cv.imshow("Binary", src_bin)
cv.imshow("Horizontal", horizontal)
cv.imshow("Vertical", vertical)
cv.imshow("Combined", mask)
cv.waitKey(0)
cv.destroyAllWindows()
