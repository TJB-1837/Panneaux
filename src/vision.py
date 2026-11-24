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


def yellow_diamond_box(hsv, polygon):
    """Accept a solid yellow diamond surrounded by a contrasting pale border."""
    if len(polygon) != 4:
        return None
    points = polygon.reshape(4, 2)
    x, y, width, height = cv2.boundingRect(points)
    normalized = (points - (x, y)) / (width - 1, height - 1)
    tips = np.array([[0.5, 0], [1, 0.5], [0.5, 1], [0, 0.5]])
    distances = np.linalg.norm(normalized[:, None] - tips[None, :], axis=2)
    if np.any(distances.min(axis=0) > 0.25):
        return None

    center = points.mean(axis=0)
    surrounding = np.rint(center + (points - center) * 2.3).astype(np.int32)
    left, top, crop_width, crop_height = cv2.boundingRect(surrounding)
    right, bottom = left + crop_width, top + crop_height
    if left < 0 or top < 0 or right > hsv.shape[1] or bottom > hsv.shape[0]:
        return None
    crop = hsv[top:bottom, left:right]
    regions = []
    # Sample the border and a separate band beyond the expected outer edge.
    for inner_scale, outer_scale in ((1.1, 1.6), (2.0, 2.3)):
        ring = np.zeros((crop_height, crop_width), dtype=np.uint8)
        outer = np.rint(center + (points - center) * outer_scale).astype(np.int32)
        inner = np.rint(center + (points - center) * inner_scale).astype(np.int32)
        cv2.fillConvexPoly(ring, outer - (left, top), 1)
        cv2.fillConvexPoly(ring, inner - (left, top), 0)
        regions.append(crop[ring > 0])
    border, background = regions
    pale = (border[:, 1] < 100) & (border[:, 2] >= 70)
    if (pale.mean() < 0.65
            or np.median(border[:, 2]) < np.median(background[:, 2]) + 15):
        return None
    outline = np.rint(center + (points - center) * 1.8).astype(np.int32)
    x, y, width, height = cv2.boundingRect(outline)
    return x, y, x + width, y + height


def color_boxes(frame, min_size, max_size):
    """Propose compact red, blue and yellow regions with a convex outline."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 100, 60), (10, 255, 255))
    red |= cv2.inRange(hsv, (170, 100, 60), (179, 255, 255))
    blue = cv2.inRange(hsv, (90, 100, 50), (135, 255, 255))
    yellow = cv2.inRange(hsv, (15, 100, 80), (35, 255, 255))
    neutral = ((hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 120)) | (hsv[:, :, 2] < 70)
    kernel = np.ones((5, 5), dtype=np.uint8)
    candidates = []
    for color, mask in (("red", red), ("blue", blue), ("yellow", yellow)):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if min(width, height) < min_size or (max_size is not None and max(width, height) > max_size):
                continue
            if not 0.5 <= width / height <= 2.0:
                continue
            if x == 0 or y == 0 or x + width >= frame.shape[1] or y + height >= frame.shape[0]:
                continue
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if area < 0.4 * width * height or area < 0.8 * hull_area:
                continue
            # Lettering and slight gaps can make a sign's color contour concave.
            polygon = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True)
            if not 3 <= len(polygon) <= 10:
                continue
            silhouette = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(silhouette, [hull - np.array([x, y])], -1, 255, -1)
            interior = silhouette > 0
            detail = neutral[y:y + height, x:x + width] & interior
            if np.count_nonzero(detail) < 0.1 * np.count_nonzero(interior):
                if color == "yellow":
                    box = yellow_diamond_box(hsv, polygon)
                    if box is not None and (max_size is None
                            or max(box[2] - box[0], box[3] - box[1]) <= max_size):
                        candidates.append(box)
                continue
            candidates.append((x, y, x + width, y + height))
    # Favor large, detailed regions when the inference budget is exhausted.
    return sorted(candidates, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)


def suppress_overlaps(boxes, threshold=0.5, limit=None):
    """Keep the first box among overlapping candidates, preserving caller order."""
    selected = []
    for box in boxes:
        if all(intersection_over_union(box, previous) < threshold for previous in selected):
            selected.append(box)
            if limit is not None and len(selected) >= limit:
                break
    return selected


def filter_predictions(predictions, threshold):
    ranked = sorted((item for item in predictions if item[2] >= threshold),
                    key=lambda item: item[2], reverse=True)
    selected = []
    for box, label, score in ranked:
        if all(intersection_over_union(box, kept[0]) < 0.5 for kept in selected):
            selected.append((box, label, score))
    return selected


def intersection_over_union(first, second):
    x1, y1, x2, y2 = first
    a1, b1, a2, b2 = second
    intersection = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
    union = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - intersection
    return intersection / union if union else 0.0
