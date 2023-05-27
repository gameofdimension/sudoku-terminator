from collections import defaultdict

import cv2
import numpy as np

from paddle_model.model import PaddleModel


def get_split_points(lines):
    n = 9
    dist = np.max(lines, axis=1)
    step = dist.shape[0] // n
    forward = np.array([step * i for i in range(n + 1)])
    backward = np.array([dist.shape[0] - 1 - i * step for i in range(n + 1)][::-1])
    init = np.clip((forward + backward) // 2, 0, dist.shape[0] - 1)

    for p in init:
        dist[p] = 255

    nonzero = np.asarray(dist > 0).nonzero()[0]

    def clustering(points, centers, it):
        def assign(px, carr):
            return np.argmin(np.abs(px - carr))

        def update_center(assignment):
            return sorted([int(np.mean(np.array(lst))) for lst in assignment.values()])

        while True:
            match = defaultdict(list)
            for x in points:
                match[assign(x, centers)].append(x)
            centers = update_center(match)
            it -= 1
            if it == 0:
                break
        return centers

    res = clustering(nonzero, init.copy().tolist(), 5)
    if len(res) < 10:
        return init.tolist()
    return res


def detect_horizontal_lines(gray):
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    return detected_lines


def get_row_splits(gray):
    tmp = gray.copy()
    lines = detect_horizontal_lines(tmp)
    return get_split_points(lines)


def get_col_splits(gray):
    tmp = gray.copy()
    tmp = cv2.rotate(tmp, cv2.ROTATE_90_CLOCKWISE)
    return get_row_splits(tmp)


def remove_horizontal_lines(gray):
    """
    https://stackoverflow.com/a/58002605
    https://docs.opencv.org/4.x/dd/dd7/tutorial_morph_lines_detection.html
    :param gray: grayscale image
    :return: image with line removed
    """
    detected_lines = detect_horizontal_lines(gray)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(gray, [c], -1, (255, 255, 255), 2)

    # Repair image
    # repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
    # result = 255 - cv2.morphologyEx(255 - gray, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

    # cv2.imshow('thresh', thresh)
    # cv2.imshow('detected_lines', detected_lines)
    # cv2.imshow('image', gray)
    # cv2.imshow('result', result)
    # cv2.waitKey()
    return gray


def remove_lines(warped):
    copy1 = warped.copy()
    for _ in range(3):
        copy1 = remove_horizontal_lines(copy1)

    copy2 = warped.copy()
    for _ in range(3):
        copy2 = cv2.rotate(copy2, cv2.ROTATE_90_CLOCKWISE)
        copy2 = remove_horizontal_lines(copy2)
        copy2 = cv2.rotate(copy2, cv2.ROTATE_90_COUNTERCLOCKWISE)

    tmp = np.maximum(copy1, warped)
    return np.maximum(copy2, tmp)


def adjust_orientation(model: PaddleModel, image: np.ndarray, gray: np.ndarray):
    direction = model.predict_orientation(img=image)
    if direction == 1:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image, gray
    if direction == 2:
        image = cv2.rotate(image, cv2.ROTATE_180)
        gray = cv2.rotate(gray, cv2.ROTATE_180)
        return image, gray
    if direction == 3:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        return image, gray
    return image, gray
