from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ORIGIN = "https://www.slavcorpora.ru"
SAMPLE_ID = "b008ae91-32cf-4d7d-84e4-996144e4edb7"

VARIANT = 11
METHOD_NAME = "Адаптивная бинаризация Феня и Тана"
WINDOW_SIZES = [3, 25]
# Параметры метода Feng–Tan.
K1 = 0.15
K2 = 0.20
GAMMA = 2.0
SECONDARY_SCALE = 3  # вторичное окно больше основного

IMAGE_INDICES = [0, 5, 10]

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results_variant11"
SRC_DIR = BASE_DIR / "src_variant11"
REPORT_PATH = BASE_DIR / "report-2-variant11.md"


@dataclass
class CaseResult:
    case_no: int
    image_index: int
    source_url: str
    width: int
    height: int
    source_name: str
    gray_name: str
    binary_names: dict[int, str]


def fetch_image_paths(origin: str, sample_id: str) -> list[str]:
    response = requests.get(f"{origin}/api/samples/{sample_id}", timeout=30)
    response.raise_for_status()
    sample_data = response.json()
    return [f"{origin}/images/{page['filename']}" for page in sample_data["pages"]]


def download_image_rgb(image_url: str) -> np.ndarray:
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    return np.asarray(pil_image, dtype=np.uint8)


def save_rgb(image: np.ndarray, path: Path) -> None:
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB").save(path)


def save_gray(image: np.ndarray, path: Path) -> None:
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="L").save(path)


def rgb_to_grayscale_weighted(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float64)
    gray = 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]
    return np.clip(gray, 0, 255).round().astype(np.uint8)


def _integral_sum(padded: np.ndarray, window_size: int, h: int, w: int) -> np.ndarray:
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    y0 = np.arange(h)[:, None]
    x0 = np.arange(w)[None, :]
    y1 = y0 + window_size
    x1 = x0 + window_size
    return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]


def _sliding_min_1d(arr: np.ndarray, window_size: int) -> np.ndarray:
    """Минимум по скользящему окну для одномерного массива за O(n)."""
    result = np.empty(len(arr) - window_size + 1, dtype=arr.dtype)
    dq: deque[int] = deque()

    for i, value in enumerate(arr):
        while dq and dq[0] <= i - window_size:
            dq.popleft()
        while dq and arr[dq[-1]] >= value:
            dq.pop()
        dq.append(i)
        if i >= window_size - 1:
            result[i - window_size + 1] = arr[dq[0]]

    return result


def local_min_fast(gray_f: np.ndarray, window_size: int) -> np.ndarray:
    """
    Быстрый локальный минимум через два одномерных прохода:
    сначала по строкам, потом по столбцам.
    """
    radius = window_size // 2
    padded = np.pad(gray_f, ((radius, radius), (radius, radius)), mode="edge")

    h, w = gray_f.shape
    row_min = np.empty((h + 2 * radius, w), dtype=gray_f.dtype)
    for y in range(h + 2 * radius):
        row_min[y, :] = _sliding_min_1d(padded[y, :], window_size)

    local_min = np.empty((h, w), dtype=gray_f.dtype)
    for x in range(w):
        local_min[:, x] = _sliding_min_1d(row_min[:, x], window_size)

    return local_min


def local_mean_std_min(gray: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if window_size % 2 == 0 or window_size < 3:
        raise ValueError("Размер окна должен быть нечетным и >= 3.")

    gray_f = gray.astype(np.float64)
    radius = window_size // 2
    padded = np.pad(gray_f, ((radius, radius), (radius, radius)), mode="edge")
    padded_sq = padded * padded

    h, w = gray.shape
    local_sum = _integral_sum(padded, window_size, h, w)
    local_sum_sq = _integral_sum(padded_sq, window_size, h, w)

    pixels_in_window = float(window_size * window_size)
    mean = local_sum / pixels_in_window
    variance = np.maximum(local_sum_sq / pixels_in_window - mean * mean, 0.0)
    std = np.sqrt(variance)
    local_min = local_min_fast(gray_f, window_size)

    return mean, std, local_min


def feng_tan_binarization(
    gray: np.ndarray,
    window_size: int,
    k1: float = K1,
    k2: float = K2,
    gamma: float = GAMMA,
    secondary_scale: int = SECONDARY_SCALE,
) -> np.ndarray:
    gray_f = gray.astype(np.float64)
    mean, std, local_min = local_mean_std_min(gray, window_size)

    secondary_window = max(window_size * secondary_scale, window_size + 2)
    if secondary_window % 2 == 0:
        secondary_window += 1

    radius = secondary_window // 2
    padded = np.pad(gray_f, ((radius, radius), (radius, radius)), mode="edge")
    padded_sq = padded * padded
    h, w = gray.shape
    local_sum = _integral_sum(padded, secondary_window, h, w)
    local_sum_sq = _integral_sum(padded_sq, secondary_window, h, w)
    pixels_in_window = float(secondary_window * secondary_window)
    secondary_mean = local_sum / pixels_in_window
    secondary_variance = np.maximum(local_sum_sq / pixels_in_window - secondary_mean * secondary_mean, 0.0)
    secondary_std = np.sqrt(secondary_variance)

    dynamic_range = np.maximum(secondary_std, 1e-6)
    ratio = np.power(std / dynamic_range, gamma)
    alpha2 = k1 * ratio
    alpha3 = k2 * ratio

    threshold = (1.0 - alpha2) * mean + alpha3 * local_min
    return np.where(gray_f > threshold, 255, 0).astype(np.uint8)


def cleanup_generated_files(directory: Path) -> None:
    if not directory.exists():
        return
    for file_path in directory.glob("img*"):
        if file_path.is_file():
            file_path.unlink()


def write_report(cases: list[CaseResult]) -> None:
    window_label = ", ".join(f"{w}x{w}" for w in WINDOW_SIZES)
    lines: list[str] = []
    lines.append("# Лабораторная работа №2")
    lines.append("## Обесцвечивание и бинаризация растровых изображений")
    lines.append("")
    lines.append(f"### Вариант {VARIANT}: {METHOD_NAME}")
    lines.append(f"### Размеры окон: {window_label}")
    lines.append("")
    lines.append("### Исходные данные")
    lines.append('- Использована выборка "Жесть" с сайта slavcorpora.ru по указанию преподавателя.')
    lines.append(f"- Количество изображений: {len(cases)}")
    lines.append(f"- Размеры окон метода: `{window_label}`")
    lines.append(f"- Параметры метода: `k1={K1}`, `k2={K2}`, `gamma={GAMMA}`, вторичное окно = `3 * D`")
    lines.append("- Исходные изображения сохранены локально в PNG, полутоновые и бинарные — в BMP.")
    lines.append("")
    lines.append("### Теоретические сведения")
    lines.append("")
    lines.append("Согласно таблице вариантов, для варианта 11 требуется реализовать адаптивную бинаризацию Феня и Тана с окнами 3×3 и 25×25.")
    lines.append("")
    lines.append("Обесцвечивание выполняется по взвешенной формуле яркости:")
    lines.append("")
    lines.append("```text")
    lines.append("I(x, y) = 0.299 * R(x, y) + 0.587 * G(x, y) + 0.114 * B(x, y)")
    lines.append("```")
    lines.append("")
    lines.append("Для метода Феня и Тана вычисляются локальные характеристики изображения в основном и вторичном окнах:")
    lines.append("")
    lines.append("```text")
    lines.append("m(x, y)  — локальное среднее в основном окне")
    lines.append("s(x, y)  — локальное среднеквадратическое отклонение в основном окне")
    lines.append("M(x, y)  — локальный минимум яркости в основном окне")
    lines.append("R(x, y)  — локальный динамический диапазон по вторичному окну")
    lines.append("")
    lines.append("alpha2(x, y) = k1 * (s(x, y) / R(x, y))^gamma")
    lines.append("alpha3(x, y) = k2 * (s(x, y) / R(x, y))^gamma")
    lines.append("T(x, y)      = (1 - alpha2(x, y)) * m(x, y) + alpha3(x, y) * M(x, y)")
    lines.append("B(x, y)      = 255, если I(x, y) > T(x, y), иначе 0")
    lines.append("```")
    lines.append("")
    lines.append("### 1. Приведение полноцветных изображений к полутоновым")
    lines.append("")

    for case in cases:
        lines.append(f"#### 1.{case.case_no} Изображение {case.case_no}")
        lines.append(f"Источник: `{case.source_url}`")
        lines.append("")
        lines.append("| Исходное (RGB, PNG) | Полутоновое (BMP) |")
        lines.append("|:-------------------:|:-----------------:|")
        lines.append(f"| ![source](src_variant11/{case.source_name}) | ![gray](src_variant11/{case.gray_name}) |")
        lines.append("")

    lines.append("### 2. Бинаризация методом Феня и Тана")
    lines.append("")
    for case in cases:
        lines.append(f"#### 2.{case.case_no} Изображение {case.case_no}")
        lines.append("")
        header = "| Полутоновое | " + " | ".join(f"Фень-Тан {w}x{w}" for w in WINDOW_SIZES) + " |"
        align = "|:-----------:|" + "|".join([":--------------:" for _ in WINDOW_SIZES]) + "|"
        cells = [f"![gray](src_variant11/{case.gray_name})"] + [
            f"![w{w}](src_variant11/{case.binary_names[w]})" for w in WINDOW_SIZES
        ]
        row = "| " + " | ".join(cells) + " |"
        lines.append(header)
        lines.append(align)
        lines.append(row)
        lines.append("")

    lines.append("### Результаты выполнения")
    lines.append("")
    lines.append("| Изображение | Размер | Бинарные файлы |")
    lines.append("|:------------|-------:|:---------------|")
    for case in cases:
        size = f"{case.width}x{case.height}"
        binary_files = ", ".join(case.binary_names[w] for w in WINDOW_SIZES)
        lines.append(f"| №{case.case_no} (индекс {case.image_index}) | {size} | `{binary_files}` |")
    lines.append("")
    lines.append("### Выводы")
    lines.append("")
    lines.append("1. Реализовано приведение полноцветного RGB-изображения к полутоновому без использования библиотечной функции grayscale.")
    lines.append("2. Для варианта 11 реализована адаптивная бинаризация Феня и Тана для окон 3x3 и 25x25 без применения библиотечных функций бинаризации.")
    lines.append("3. На примерах из выборки \"Жесть\" видно, что малое окно лучше подчеркивает мелкие детали, а большое окно сильнее сглаживает фон и неравномерность освещения.")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("Создаём папки...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_generated_files(RESULTS_DIR)
    cleanup_generated_files(SRC_DIR)

    print("Получаем список изображений из выборки...")
    image_paths = fetch_image_paths(ORIGIN, SAMPLE_ID)
    if not image_paths:
        raise RuntimeError("Список изображений пуст.")
    print(f"Найдено изображений: {len(image_paths)}")

    cases: list[CaseResult] = []
    for case_no, image_index in enumerate(IMAGE_INDICES, start=1):
        if image_index < 0 or image_index >= len(image_paths):
            raise IndexError(f"Индекс {image_index} выходит за пределы списка image_paths.")

        print(f"\nОбрабатывается изображение {case_no} (индекс {image_index})...")
        source_url = image_paths[image_index]
        source_rgb = download_image_rgb(source_url)
        gray = rgb_to_grayscale_weighted(source_rgb)

        source_name = f"img{case_no}_source.png"
        gray_name = f"img{case_no}_grayscale.bmp"
        binary_names: dict[int, str] = {}

        save_rgb(source_rgb, RESULTS_DIR / source_name)
        save_gray(gray, RESULTS_DIR / gray_name)
        save_rgb(source_rgb, SRC_DIR / source_name)
        save_gray(gray, SRC_DIR / gray_name)

        for window_size in WINDOW_SIZES:
            print(f"  Бинаризация методом Феня-Тана, окно {window_size}x{window_size}...")
            binary = feng_tan_binarization(gray, window_size=window_size)
            binary_name = f"img{case_no}_binary_feng_tan_w{window_size}.bmp"
            binary_names[window_size] = binary_name

            save_gray(binary, RESULTS_DIR / binary_name)
            save_gray(binary, SRC_DIR / binary_name)

        height, width = gray.shape
        cases.append(
            CaseResult(
                case_no=case_no,
                image_index=image_index,
                source_url=source_url,
                width=width,
                height=height,
                source_name=source_name,
                gray_name=gray_name,
                binary_names=binary_names,
            )
        )

    print("\nФормируем отчёт...")
    write_report(cases)

    print("\nЛабораторная работа №2 выполнена.")
    print(f"Вариант: {VARIANT} ({METHOD_NAME})")
    print(f"Окна: {', '.join(f'{w}x{w}' for w in WINDOW_SIZES)}")
    print(f"Результаты: {RESULTS_DIR}")
    print(f"Файлы для отчёта: {SRC_DIR}")
    print(f"Отчёт: {REPORT_PATH}")


if __name__ == "__main__":
    main()
