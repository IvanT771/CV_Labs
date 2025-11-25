import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# ============================================================
# Пути под твой проект
# ============================================================

IMAGES_DIR = Path("PR/Images")     # исходные фото машин
TEMPLATES_DIR = Path("PR/gosznak") # шаблоны ГОСТ (буквы+цифры)
RESULTS_DIR = Path("PR/Results")   # сюда пишем итоговые картинки
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Формат номера: буква, 3 цифры, 2 буквы, 2 цифры региона
# Всего 8 символов
# ============================================================

CHAR_PATTERN: List[str] = [
    "letter", "digit", "digit", "digit",
    "letter", "letter",
    "digit", "digit",
]

# Координаты сегментации символов внутри ROI номера
# Эти координаты взяты из кода репозитория cv1 (split_number_by_image),
# рассчитаны под типичный ROI номерного знака.
CHAR_BOXES: List[Tuple[int, int, int, int]] = [
    (7, 4, 15, 20),
    (20, 0, 18, 30),
    (35, 3, 15, 21),
    (47, 0, 15, 30),
    (62, 5, 18, 19),
    (75, 5, 15, 19),
    (90, 0, 15, 20),
    (100, 0, 15, 20),
]


# ============================================================
# Коррекция яркости (из cv1)
# ============================================================

def adjust_brightness(image: np.ndarray) -> np.ndarray:
    """
    Выравнивание яркости в YUV, как в репозитории cv1.
    """
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


# ============================================================
# Поиск ROI номерного знака (адаптация кода из cv1)
# ============================================================

def find_best_plate_roi(image: np.ndarray) -> np.ndarray | None:
    """
    Находит лучший ROI номерного знака в исходном цветном изображении.
    Логика взята из репозитория cv1 (пороговый перебор + minAreaRect + solidity).
    Возвращает цветной ROI (BGR) или None.
    """
    filtered_img = cv2.GaussianBlur(image, (9, 9), 0)
    gray_image = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)

    # В оригинальном коде был список [50, 55, ..., 255]
    threshold_values = list(range(50, 256, 5))

    best_roi = None
    best_solidity = 0.0

    for thrs in threshold_values:
        _, binary_image = cv2.threshold(
            gray_image, thrs, 255, cv2.THRESH_BINARY
        )

        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            rect = cv2.minAreaRect(contour)
            (x, y), (w, h), angle = rect

            # Отбрасываем странные углы, как в cv1
            if 15 < angle < 45:
                continue

            if w == 0 or h == 0:
                continue

            aspect_ratio = w / h if w > h else h / w
            area = w * h

            # Фильтр по соотношению сторон и площади (как в cv1, с лёгким запасом)
            if not (2.5 < aspect_ratio < 5.8 and 1500 < area < 10000):
                continue

            contour_area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(contour_area) / hull_area if hull_area > 0 else 0

            # Фильтр по "заполненности" контура
            if solidity < 0.8:
                continue

            if solidity > best_solidity:
                best_solidity = solidity

                # Коррекция угла, как в cv1
                angle_corr = angle if angle < 40 else angle - 90
                M = cv2.getRotationMatrix2D((x, y), angle_corr, 1.0)
                rotated = cv2.warpAffine(
                    image,
                    M,
                    (image.shape[1], image.shape[0])
                )

                # Вычисление размеров ROI
                vertical_value = h if w > h else w
                horizontal_value = w if w > h else h
                horizontal_value = min(horizontal_value, 120)

                y1 = int(y - vertical_value / 2)
                y2 = int(y + vertical_value / 2)
                x1 = int(x - horizontal_value / 2)
                x2 = int(x + horizontal_value / 2)

                # Ограничиваем в пределах изображения
                h_img, w_img = image.shape[:2]
                y1 = max(y1, 0)
                y2 = min(y2, h_img)
                x1 = max(x1, 0)
                x2 = min(x2, w_img)

                if y2 > y1 and x2 > x1:
                    best_roi = rotated[y1:y2, x1:x2]

    return best_roi


# ============================================================
# Нарезка символов по фиксированным координатам (как в cv1)
# ============================================================

def split_number_by_image(image: np.ndarray) -> List[np.ndarray]:
    """
    Вырезание 8 символов из ROI номерного знака.
    Координаты взяты из split_number_by_image из репозитория cv1.
    Ожидается, что ROI по размеру близок к тем, для которых
    подбирались эти координаты.
    """
    symbols: List[np.ndarray] = []
    for (x, y, w, h) in CHAR_BOXES:
        symbol = image[y:y + h, x:x + w]
        symbols.append(symbol)
    return symbols


# ============================================================
# Бинаризация отдельного символа (из cv1)
# ============================================================

def binaryzation_number_symbol(symbol_image: np.ndarray) -> np.ndarray:
    """
    Преобразование символа в двоичное изображение:
    - в оттенки серого
    - выравнивание гистограммы
    - гауссово размытие
    - адаптивная бинаризация
    """
    grayscale = cv2.cvtColor(symbol_image, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.equalizeHist(grayscale)
    grayscale = cv2.GaussianBlur(grayscale, (3, 3), 0)

    binary_image = cv2.adaptiveThreshold(
        grayscale,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    return binary_image


# ============================================================
# Сравнение символа с шаблонами (адаптация compare_function)
# ============================================================

def compare_symbol_with_templates(
    symbol_image: np.ndarray,
    templates_folder: Path,
    is_digit: bool,
    is_region_digit: bool,
) -> str:
    """
    Сравнение одного символа с набором шаблонов из папки templates_folder.
    Логика из compare_function в cv1, адаптирована под PR/gosznak.
    """
    binary_image = binaryzation_number_symbol(symbol_image)

    # Загружаем имена файлов шаблонов
    templates = [
        f
        for f in os.listdir(templates_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Отбор по типу символа: цифра / буква
    if is_digit:
        templates = [f for f in templates if f[0].isdigit()]
    else:
        templates = [f for f in templates if not f[0].isdigit()]

    best_match = "?"
    best_score = -1.0

    for template_name in templates:
        template_path = templates_folder / template_name
        template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue

        # Масштабный коэффициент — взят из исходного кода cv1
        if is_region_digit:
            scale_factor = 0.04285714285  # цифры региона
        else:
            # для цифр номера и букв
            scale_factor = 0.0488505747126437 if is_digit else 0.0599078341013825

        new_width = int(template.shape[1] * scale_factor)
        new_height = int(template.shape[0] * scale_factor)
        if new_width <= 0 or new_height <= 0:
            continue

        template_resized = cv2.resize(template, (new_width, new_height))

        # matchTemplate
        res = cv2.matchTemplate(
            binary_image, template_resized, cv2.TM_CCOEFF_NORMED
        )
        max_val = cv2.minMaxLoc(res)[1]

        if max_val > best_score:
            best_score = max_val
            # имя файла без расширения берём как символ
            best_match = os.path.splitext(template_name)[0]

    return best_match


# ============================================================
# Распознавание номера по ROI
# ============================================================

def recognize_number_from_roi(roi_bgr: np.ndarray) -> str:
    """
    Полный цикл распознавания номера по вырезанному ROI:
    - корректировка яркости
    - нарезка на 8 символов
    - сравнение с шаблонами ГОСТ
    """
    roi_bgr = adjust_brightness(roi_bgr)

    # Вырезаем символы
    symbols = split_number_by_image(roi_bgr)

    recognized_number = ""

    for i, symbol in enumerate(symbols):
        is_digit = i in [1, 2, 3, 6, 7]      # 3 цифры номера + 2 цифры региона
        is_region_digit = i in [6, 7]        # последние 2 — регион
        ch = compare_symbol_with_templates(
            symbol,
            TEMPLATES_DIR,
            is_digit=is_digit,
            is_region_digit=is_region_digit,
        )
        recognized_number += ch if ch else "?"

    # Формируем строку вида "А123БВ 77"
    if len(recognized_number) == 8:
        return recognized_number[:6] + " " + recognized_number[6:]
    return recognized_number


# ============================================================
# Обработка одного исходного изображения
# ============================================================

def process_image(image_path: Path) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        print("Не удалось загрузить:", image_path)
        return

    roi = find_best_plate_roi(img)
    if roi is None:
        print(f"{image_path.name}: не найден номерной знак.")
        return

    plate_text = recognize_number_from_roi(roi)
    print(f"{image_path.name}: {plate_text}")

    # Финальная визуализация: исходное + текст + миниатюра ROI
    out = img.copy()

    cv2.putText(
        out,
        plate_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    # Вставляем ROI в правый верхний угол
    ph, pw = roi.shape[:2]
    scale = 0.7  # чуть уменьшим
    roi_small = cv2.resize(roi, (int(pw * scale), int(ph * scale)))
    rh, rw = roi_small.shape[:2]
    H, W = out.shape[:2]

    y1, y2 = 50, 50 + rh
    x1, x2 = W - rw - 20, W - 20
    if y2 <= H and x1 >= 0:
        out[y1:y2, x1:x2] = roi_small

    out_path = RESULTS_DIR / f"result_{image_path.name}"
    cv2.imwrite(str(out_path), out)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    images = sorted(IMAGES_DIR.glob("*.jpg"))
    if not images:
        print("Нет изображений в", IMAGES_DIR)
        return

    for img_path in images:
        process_image(img_path)

    print("Готово. Результаты в", RESULTS_DIR)


if __name__ == "__main__":
    main()
