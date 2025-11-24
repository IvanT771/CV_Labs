import cv2 as cv

# === ЛР10_2. BRIEF — бинарный дескриптор признаков
# === https://docs.opencv.org/3.4/dc/d7d/tutorial_py_brief.html
#
# pip install opencv-contrib-python

# 1. Проверяем наличие модуля xfeatures2d (contrib)
if not hasattr(cv, "xfeatures2d"):
    print("Модуль cv.xfeatures2d недоступен.")
    print("Для примера BRIEF установите пакет 'opencv-contrib-python'.")
else:
    # 2. Загружаем изображение (градации серого)
    img = cv.imread('Lab3/img2.jpg', cv.IMREAD_GRAYSCALE)
    assert img is not None, "Файл Lab3/img2.jpg не найден."

    # 3. Создаём детектор ключевых точек (STAR) и дескриптор BRIEF
    #    как в примере документации
    star = cv.xfeatures2d.StarDetector_create()
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

    # 4. Детектируем ключевые точки и вычисляем дескрипторы
    keypoints = star.detect(img, None)
    keypoints, descriptors = brief.compute(img, keypoints)

    print("Найдено ключевых точек:", len(keypoints))
    print("Форма матрицы дескрипторов:", descriptors.shape)

    # 5. Отобразим ключевые точки на изображении
    # img_kp = cv.drawKeypoints(img, keypoints, None, color=(0, 255, 0))
    # cv.imshow('BRIEF keypoints', img_kp)
    cv.waitKey(0)
    cv.destroyAllWindows()

