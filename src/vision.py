import cv2
import numpy as np

from .data import Box, clip_box


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image illisible : {path}")
    return image


def crop_input(frame, box: Box, width: int, height: int, channels: int):
    x1, y1, x2, y2 = box
    pad_x, pad_y = (x2 - x1) * 0.08, (y2 - y1) * 0.08
    x1, y1, x2, y2 = clip_box(
        (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), frame.shape[1], frame.shape[0]
    )
    crop = frame[y1:y2, x1:x2]
    color_mode = cv2.COLOR_BGR2GRAY if channels == 1 else cv2.COLOR_BGR2RGB
    crop = cv2.cvtColor(crop, color_mode)
    scale = min(width / crop.shape[1], height / crop.shape[0])
    scaled_width = min(width, max(1, round(crop.shape[1] * scale)))
    scaled_height = min(height, max(1, round(crop.shape[0] * scale)))
    resized = cv2.resize(
        crop, (scaled_width, scaled_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    ).reshape(scaled_height, scaled_width, channels)
    result = np.full((height, width, channels), 114, dtype=np.uint8)
    x, y = (width - scaled_width) // 2, (height - scaled_height) // 2
    result[y:y + scaled_height, x:x + scaled_width] = resized
    return result


def canny_boxes(frame, min_size, max_size, low, high):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, low, high)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if min_size <= min(width, height) and (max_size is None or max(width, height) <= max_size):
            yield x, y, x + width, y + height


def intersection_over_union(first, second):
    x1, y1, x2, y2 = first
    a1, b1, a2, b2 = second
    intersection = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
    union = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - intersection
    return intersection / union if union else 0.0
