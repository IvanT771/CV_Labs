#Замер производительности
import cv2 as cv
import timeit
import numpy as np

e1 = cv.getTickCount()
# your code execution
e2 = cv.getTickCount()
time = (e2 - e1)/ cv.getTickFrequency()

img1 = cv.imread('Lab2/open-cv-logo.jpg')
assert img1 is not None, "file could not be read, check with os.path.exists()"

e1 = cv.getTickCount()
for i in range(5,49,2):
    img1 = cv.medianBlur(img1,i)
e2 = cv.getTickCount()

t = (e2 - e1)/cv.getTickFrequency()
print(t)

# check if optimization is enabled
print(cv.useOptimized())

# Проверяем, включены ли оптимизации
print("Optimized:", cv.useOptimized())

# Тест с включёнными оптимизациями
start = cv.getTickCount()
for _ in range(10):
    res = cv.medianBlur(img1, 49)
end = cv.getTickCount()
print("With optimization:", (end - start) /cv.getTickFrequency(), "ms per loop")

# Выключаем оптимизации
cv.setUseOptimized(False)
print("Optimized:", cv.useOptimized())

# Тест без оптимизаций
start = cv.getTickCount()
for _ in range(10):
    res = cv.medianBlur(img1, 49)
end = cv.getTickCount()
print("Without optimization:", (end - start) /cv.getTickFrequency(), "ms per loop")

# Включаем обратно
cv.setUseOptimized(True)

# --- 1. Сравнение операций возведения в квадрат ---
x = 5
print("Скалярные операции Python:")
print("x ** 2:", timeit.timeit('y = x ** 2', number=10_000_000, globals=globals()))
print("x * x:", timeit.timeit('y = x * x', number=10_000_000, globals=globals()))

z = np.uint8([5])
print("\nОперации NumPy:")
print("z * z:", timeit.timeit('y = z * z', number=1_000_000, globals=globals()))
print("np.square(z):", timeit.timeit('y = np.square(z)', number=1_000_000, globals=globals()))

# --- 2. Сравнение функций OpenCV и NumPy ---
img = np.random.randint(0, 2, (1000, 1000), dtype=np.uint8)

print("\nПодсчёт ненулевых элементов:")
print("cv.countNonZero:", timeit.timeit('cv.countNonZero(img)', number=100_000, globals=globals()))
print("np.count_nonzero:", timeit.timeit('np.count_nonzero(img)', number=1_000, globals=globals()))