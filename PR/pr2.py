from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

IMAGES_DIR = Path("PR/Images")
TEMPLATES_DIR = Path("PR/gosznak")

def load_image(imagePath: Path):
    """
    Загрузка и подготовка изображения для обработки.

    Аргументы:
    image_path (str): Путь к изображению.

    Возвращает:
    image (numpy.ndarray): Загруженное изображение.

    Исключения:
    ValueError: Если изображение не удалось загрузить.
    """
    image = cv.imread(str(imagePath))
    if image is None:
        raise ValueError("Could not load image")
    return image

# 1. Собираем список изображений
images = sorted(IMAGES_DIR.glob("*.jpg"))

for image_path in images:
    loaded_image = load_image(image_path)
 # grayscale
    gray = cv.cvtColor(loaded_image, cv.COLOR_BGR2GRAY)

    # median blur
    median_filtered = cv.medianBlur(gray, 1)

    # binarization
    binarized = cv.adaptiveThreshold(
        median_filtered,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,  # или cv.ADAPTIVE_THRESH_MEAN_C
        cv.THRESH_BINARY,
        25,   # размер окна
        10    # константа
    )

    # вывод
    plt.figure(figsize=(6, 6))
    plt.title(str(image_path))
    plt.imshow(binarized, cmap="gray")
    plt.axis("off")

plt.show()
