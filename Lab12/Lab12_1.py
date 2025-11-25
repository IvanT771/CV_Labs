import cv2 as cv

# === ЛР12_1. Обнаружение лиц каскадными классификаторами
# === По примеру:
# === https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

# 1. Загружаем каскады
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

assert not face_cascade.empty(), "Не удалось загрузить каскад лиц."
assert not eye_cascade.empty(),  "Не удалось загрузить каскад глаз."

# 2. Загружаем изображение
img = cv.imread("Lab12/face.jpg")  
if img is None:
    raise SystemExit("Не удалось загрузить изображение.")

# 3. Преобразуем в оттенки серого
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 4. Обнаруживаем лица
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    roi_gray  = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

# 5. Отображение результата
cv.imshow("Face & Eye detection", img)
cv.waitKey(0)
cv.destroyAllWindows()
