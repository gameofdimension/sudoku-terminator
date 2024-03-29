import argparse

import cv2
from loguru import logger

from lucky.ocr_solver import StructuralSolver

parser = argparse.ArgumentParser(description='solve sudoku in the image')
parser.add_argument('--image-path', help='path to sudoku image', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    solver = StructuralSolver()
    image_path = args.image_path
    result = solver.run(image_path)
    if result is None:
        logger.error(f"solving puzzle fail")
    else:
        window_name = "sudoku solution"
        cv2.imshow(window_name, result)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(0)
