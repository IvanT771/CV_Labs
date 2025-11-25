import cv2 as cv

# === ЛР12_1. Обнаружение лиц каскадными классификаторами
# === По примеру из:
# === https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
#
# Вариант из документации использует изображения и/или камеру.
# Здесь используем веб‑камеру, как и в предыдущих лабораторных.

# 1. Загружаем предобученные каскады из поставки OpenCV
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

assert not face_cascade.empty(), "Не удалось загрузить каскад лиц."
assert not eye_cascade.empty(), "Не удалось загрузить каскад глаз."

# 2. Открываем камеру
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Не удалось открыть камеру.")
    raise SystemExit

print("Нажмите 'q' для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Кадр не получен, завершаем.")
        break

    # 3. Переводим кадр в оттенки серого
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 4. Обнаруживаем лица
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Рисуем прямоугольник вокруг лица
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Область интереса для глаз
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv.imshow('Face & Eye detection', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

