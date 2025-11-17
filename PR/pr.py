import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2 as cv
import numpy as np


# === Конфигурация ===

# Ожидаемый формат номера: буква, 3 цифры, 2 буквы, 2 цифры региона
CHAR_PATTERN: List[str] = [
    "letter",
    "digit",
    "digit",
    "digit",
    "letter",
    "letter",
    "digit",
    "digit",
]

IMAGES_DIR = Path("PR/Images")
TEMPLATES_DIR = Path("PR/gosznak")
RESULTS_DIR = Path("PR/Results")
RESULTS_DIR.mkdir(exist_ok=True)


def _order_box_points(pts: np.ndarray) -> np.ndarray:
    """
    Упорядочиваем 4 точки прямоугольника в виде:
    top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def load_templates(
    templates_dir: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Загружаем шаблоны символов ГОСТ.

    Возвращает два словаря:
      - digits: { '0': img, ..., '9': img }
      - letters: { 'A': img, 'B': img, ... }

    Все изображения переводятся в оттенки серого и бинаризуются.
    """
    digits: Dict[str, np.ndarray] = {}
    letters: Dict[str, np.ndarray] = {}

    for path in sorted(templates_dir.glob("*.png")):
        name = path.stem  # '0', 'A', 'B', ...
        img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # Бинаризация шаблона (Otsu)
        _, bin_img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # Делаем так, чтобы символы были белыми на чёрном фоне
        if cv.countNonZero(bin_img) > bin_img.size / 2:
            bin_img = cv.bitwise_not(bin_img)

        # У цифры '8' исходный шаблон получается инверсным,
        # поэтому дополнительно инвертируем её вручную.
        if name == "8":
            bin_img = cv.bitwise_not(bin_img)

        if name.isdigit():
            digits[name] = bin_img
        else:
            letters[name.upper()] = bin_img

    # Визуализируем все обработанные шаблоны в одном окне
    try:
        import matplotlib.pyplot as plt

        images: List[np.ndarray] = []
        labels: List[str] = []

        for ch, img in sorted(digits.items()):
            images.append(img)
            labels.append(ch)
        for ch, img in sorted(letters.items()):
            images.append(img)
            labels.append(ch)

        if images:
            cols = 8
            rows = int(np.ceil(len(images) / cols))
            plt.figure("Шаблоны ГОСТ после обработки", figsize=(2 * cols, 2 * rows))
            for i, (im, lab) in enumerate(zip(images, labels), start=1):
                ax = plt.subplot(rows, cols, i)
                ax.imshow(im, cmap="gray")
                ax.set_title(lab)
                ax.axis("off")
            plt.tight_layout()
            plt.show()
    except Exception:
        # Если matplotlib не установлен или возникла ошибка при отображении,
        # просто пропускаем визуализацию шаблонов.
        pass

    return digits, letters


def global_threshold_from_first_image(imagePath: Path) -> int:
    """
    Вычисляем общий порог для бинаризации по первой картинке с помощью метода Отсу.
    Этот порог затем применяем ко всем изображениям (требование 1 этапа).
    """
    first = imagePath
    img = cv.imread(str(first))
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить {first}")

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_blur = cv.GaussianBlur(gray, (5, 5), 0)
    # THRESH_BINARY + OTSU вернёт значение порога в ret
    ret, _ = cv.threshold(gray_blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    print(f"Глобальный порог (Отсу) по {first.name}: {ret}")
    return int(ret)


def auto_threshold(image: np.ndarray) -> np.ndarray:
    """
    Бинаризация цветного изображения с автоматическим порогом (метод Отсу).
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv.threshold(
        blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
    )
    return binary


def binarize_image(gray: np.ndarray, thr: int) -> np.ndarray:
    """
    Бинаризация изображения с заданным порогом.
    При необходимости инвертируем, чтобы номер был белым на чёрном.
    """
    _, bw = cv.threshold(gray, thr, 255, cv.THRESH_BINARY)

    # Если фон оказался белым (белых пикселей слишком много) — инвертируем
    if cv.countNonZero(bw) > bw.size / 2:
        bw = cv.bitwise_not(bw)

    return bw


def find_plate_contour(bw: np.ndarray) -> np.ndarray:
    """
    По бинарному изображению ищем контур номерного знака.

    Фильтрация по площади и соотношению сторон.
    В итоге выбираем один лучший контур.
    """
    contours, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Пороговые значения площади и соотношения сторон
    # взяты по аналогии с заготовкой из практической работы.
    min_area = 1000
    max_area = 10000
    min_aspect, max_aspect = 2.0, 5.0

    best_cnt = None
    best_area = 0

    for cnt in contours:
        x, y, cw, ch = cv.boundingRect(cnt)
        if cw == 0 or ch == 0:
            continue

        area = cw * ch
        if area < min_area or area > max_area:
            continue

        aspect = cw / float(ch)
        if not (min_aspect <= aspect <= max_aspect):
            continue

        # Выбираем контур с максимальной площадью среди подходящих
        if area > best_area:
            best_area = area
            best_cnt = cnt

    if best_cnt is None:
        raise RuntimeError("Не удалось найти контур номерного знака.")

    return best_cnt


def crop_plate(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """
    Вырезаем номерной знак по минимальному описанному прямоугольнику (minAreaRect)
    с помощью перспективного преобразования.

    Здесь мы не делаем «ручного» поворота, а используем четыре вершины
    прямоугольника и классический алгоритм выпрямления: точки упорядочиваются
    как TL, TR, BR, BL, далее строится матрица cv.getPerspectiveTransform.
    """
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)  # (4, 2)
    box = np.array(box, dtype=np.float32)

    # Упорядочиваем вершины прямоугольника: TL, TR, BR, BL
    ordered = _order_box_points(box)
    (tl, tr, br, bl) = ordered

    # Оцениваем размеры выпрямлённого прямоугольника
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    if maxWidth <= 0 or maxHeight <= 0:
        raise RuntimeError("Размеры вырезанного номерного знака равны нулю.")

    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ],
        dtype=np.float32,
    )

    M = cv.getPerspectiveTransform(ordered, dst)
    plate = cv.warpPerspective(image, M, (maxWidth, maxHeight))

    # Нормализуем ориентацию: ширина должна быть больше высоты
    if plate.shape[0] > plate.shape[1]:
        plate = cv.rotate(plate, cv.ROTATE_90_CLOCKWISE)

    # Увеличиваем вырезанный номер до комфортного размера,
    # чтобы цифры были крупными и лучше поддавались бинаризации/сопоставлению.
    h_p, w_p = plate.shape[:2]
    target_height = 120  # более высокое целевое разрешение по высоте
    if h_p < target_height:
        scale = target_height / float(h_p)
        new_w = int(w_p * scale)
        plate = cv.resize(plate, (new_w, target_height), interpolation=cv.INTER_CUBIC)

    return plate

CHAR_SPLIT_FRACTIONS = [
    0.05,
    1.4 / 8.0,
    2.4 / 8.0,
    3.1 / 8.0,
    4.0 / 8.0,
    5.1 / 8.0,
    6.0 / 8.0,
    7.2 / 8.0,
    1.0,
]

def segment_characters(
    plate_img: np.ndarray, expected_chars: int = 8
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    Нарезка номерного знака на отдельные символы.

    Здесь мы предварительно усиливаем читаемость символов:
      - переводим номер в оттенки серого,
      - подавляем шум медианной фильтрацией,
      - усиливаем контраст equalizeHist,
      - после этого режем и бинаризуем символы.
    """
    # Перевод в оттенки серого
    gray_orig = cv.cvtColor(plate_img, cv.COLOR_BGR2GRAY)

    # Очистка от шума перед нарезкой: более сильная медианная фильтрация
    gray_denoised = cv.medianBlur(gray_orig, 5)

    # Усиление контраста: CLAHE работает лучше на локальных участках,
    # чем глобальное equalizeHist, что делает цифры более чёткими.
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray_denoised)

    h, w = gray.shape

    # Захватываем центральную по вертикали полосу, чтобы
    # отсечь верхнюю/нижнюю рамку номера.
    y_top = int(0.1 * h)
    y_bottom = int(0.9 * h)

    char_images: List[np.ndarray] = []
    regions: List[Tuple[int, int, int, int]] = []

    # Жёстко делим номерной знак по заданным долям ширины
    for i in range(expected_chars):
        x_start = int(CHAR_SPLIT_FRACTIONS[i] * w)
        x_end = int(CHAR_SPLIT_FRACTIONS[i + 1] * w)

        roi_gray = gray[y_top:y_bottom, x_start:x_end]

        # Бинаризация символа с инверсией, чтобы символ был белым
        _, roi_bw = cv.threshold(
            roi_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
        )

        char_images.append(roi_bw)
        regions.append((x_start, y_top, x_end - x_start, y_bottom - y_top))

    # Визуализация самого номерного знака и нарезанных символов
    # до и после бинаризации (для отладки)
    try:
        import matplotlib.pyplot as plt

        # Отдельное окно: номер до и после обработки
        plt.figure("Номерной знак до/после обработки", figsize=(8, 3))
        ax_a = plt.subplot(1, 3, 1)
        ax_a.imshow(gray_orig, cmap="gray")
        ax_a.set_title("original")
        ax_a.axis("off")
        ax_b = plt.subplot(1, 3, 2)
        ax_b.imshow(gray_denoised, cmap="gray")
        ax_b.set_title("median")
        ax_b.axis("off")
        ax_c = plt.subplot(1, 3, 3)
        ax_c.imshow(gray, cmap="gray")
        ax_c.set_title("equalized")
        ax_c.axis("off")
        plt.tight_layout()

        cols = expected_chars
        rows = 3
        plt.figure("Нарезанные символы номерного знака", figsize=(2 * cols, 6))

        # Первая строка — исходный номерной знак (целиком)
        ax0 = plt.subplot(rows, cols, 1)
        ax0.imshow(cv.cvtColor(plate_img, cv.COLOR_BGR2RGB))
        ax0.set_title("plate")
        ax0.axis("off")

        for i in range(expected_chars):
            x_start = int(CHAR_SPLIT_FRACTIONS[i] * w)
            x_end = int(CHAR_SPLIT_FRACTIONS[i + 1] * w)
            roi_gray_show = gray[y_top:y_bottom, x_start:x_end]

            # Строка 2: исходные (серые, уже обработанные) фрагменты
            ax1 = plt.subplot(rows, cols, cols + i + 1)
            ax1.imshow(roi_gray_show, cmap="gray")
            ax1.set_title(f"{i+1} raw")
            ax1.axis("off")

            # Строка 3: бинаризованные фрагменты
            ax2 = plt.subplot(rows, cols, 2 * cols + i + 1)
            ax2.imshow(char_images[i], cmap="gray")
            ax2.set_title(f"{i+1} bin")
            ax2.axis("off")

        plt.tight_layout()
        plt.show()
    except Exception:
        # Если matplotlib недоступен, просто не показываем символы
        pass

    return char_images, regions


def match_symbol(
    symbol_img: np.ndarray, templates: Dict[str, np.ndarray]
) -> Tuple[str, float]:
    """
    Сравнение символа с набором шаблонов через cv.matchTemplate.
    Возвращает лучший символ и его вес (коэффициент корреляции).
    """
    best_char = "?"
    best_score = -1.0

    # Обеспечим тип и диапазон
    symbol = symbol_img.astype(np.uint8)

    for char, templ in templates.items():
        # Масштабируем шаблон под размер символа номерного знака
        resized_templ = cv.resize(templ, (symbol.shape[1], symbol.shape[0]))

        res = cv.matchTemplate(symbol, resized_templ, cv.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv.minMaxLoc(res)

        if max_val > best_score:
            best_score = max_val
            best_char = char

    return best_char, best_score


def recognize_plate(
    plate_img: np.ndarray,
    digits_templates: Dict[str, np.ndarray],
    letters_templates: Dict[str, np.ndarray],
) -> Tuple[str, List[Tuple[int, int, int, int]]]:
    """
    Распознаёт номерной знак по вырезанному изображению номера.

    Возвращает строку распознанного номера и список координат символов.
    """
    chars, regions = segment_characters(plate_img, expected_chars=len(CHAR_PATTERN))

    result_chars: List[str] = []

    for idx, (char_img, char_type) in enumerate(zip(chars, CHAR_PATTERN)):
        if char_type == "digit":
            char, score = match_symbol(char_img, digits_templates)
        else:
            char, score = match_symbol(char_img, letters_templates)

        result_chars.append(char)

    # Добавим пробел перед регионом для читабельности
    if len(result_chars) == 8:
        plate_text = (
            "".join(result_chars[:6]) + " " + "".join(result_chars[6:])
        )
    else:
        plate_text = "".join(result_chars)

    return plate_text, regions


def process_image(
    image_path: Path,
    thr: int,
    digits_templates: Dict[str, np.ndarray],
    letters_templates: Dict[str, np.ndarray],
) -> None:
    """
    Полная обработка одной картинки:
      - бинаризация,
      - поиск контура номера,
      - вырезание и распознавание,
      - вывод результата и сохранение картинки с подписью.
    """
    img = cv.imread(str(image_path))
    if img is None:
        print(f"Не удалось загрузить {image_path}")
        return

    orig = img.copy()

    # Для поиска номера используем бинаризацию с автоматическим порогом (Отсу)
    # по всему изображению (см. auto_threshold).
    h_img, w_img = img.shape[:2]
    bw = auto_threshold(img)

    plate_contour = find_plate_contour(bw)
    plate_img = crop_plate(img, plate_contour)
    
    plate_text, char_regions = recognize_plate(plate_img, digits_templates, letters_templates)


    print(f"{image_path.name}: распознан номер -> {plate_text}")

    # Рисуем найденный контур на исходной картинке
    x, y, w, h = cv.boundingRect(plate_contour)
    cv.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Добавляем текст номера на исходное изображение
    h_img, w_img = orig.shape[:2]
    cv.putText(
        orig,
        plate_text,
        (10, h_img - 20),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv.LINE_AA,
    )

    # Дополнительно создадим "коллаж": слева исходная, справа вырезанный номер.
    # При масштабировании сохраняем исходное соотношение сторон номера.
    ph_orig, pw_orig = plate_img.shape[:2]
    max_w = w_img // 3
    max_h = h_img // 3
    scale = min(max_w / float(pw_orig), max_h / float(ph_orig))
    new_w = max(1, int(pw_orig * scale))
    new_h = max(1, int(ph_orig * scale))
    plate_resized = cv.resize(plate_img, (new_w, new_h))
    canvas = orig.copy()
    ph, pw = plate_resized.shape[:2]
    canvas[10 : 10 + ph, w_img - pw - 10 : w_img - 10] = plate_resized

    out_path = RESULTS_DIR / f"result_{image_path.name}"
    cv.imwrite(str(out_path), canvas)


def main() -> None:
    # 1. Собираем список изображений
    images = sorted(IMAGES_DIR.glob("*.jpg"))
    if not images:
        print(f"Не найдены картинки в {IMAGES_DIR}")
        return

    # 2. Глобальный порог по первой картинке 
   

    # 3. Загружаем шаблоны ГОСТ
    digits_templates, letters_templates = load_templates(TEMPLATES_DIR)
    if not digits_templates or not letters_templates:
        print("Не удалось загрузить шаблоны ГОСТ.")
        return

    # 4. Обработка всех картинок
    for img_path in images:
        thr = global_threshold_from_first_image(img_path)
        process_image(img_path, thr, digits_templates, letters_templates)

    print(f"Результаты сохранены в {RESULTS_DIR}")


if __name__ == "__main__":
    main()
