import string
from math import floor

import cv2
import imutils
import numpy as np
from loguru import logger

from lucky.helper import adjust_orientation
from paddle_model.model import PaddleModel
from pyimagesearch.puzzle import (
    parse_image,
    build_solver,
    draw_back,
    robust_find_puzzle
)


class StructuralSolver:

    def __init__(self):
        self._paddle = PaddleModel()

    @staticmethod
    def make_cell_locs(gray):
        rs, cs = gray.shape[0] // 9, gray.shape[1] // 9
        res = []
        for y in range(0, 9):
            row = []
            for x in range(0, 9):
                start_x, end_x = cs * x, cs * (x + 1)
                start_y, end_y = rs * y, rs * (y + 1)
                row.append((start_x, start_y, end_x, end_y))
            res.append(row)
        return res

    def detect_board(self, warped: np.ndarray, debug: bool):
        lst = self._paddle.predict_image(warped)
        center = [(np.mean(mat, axis=0), cs) for mat, (cs, confidence) in lst if confidence > 0.4]
        digits = [(p, tx) for p, tx in center if len(tx) == 1 and tx in string.digits + "sS"]
        non_digit_num = len(lst) - len(digits)
        if non_digit_num > 3:
            logger.error(f"too many non digits {non_digit_num}")
            return None
        if len(digits) < 17:
            logger.error(f"too few digits {len(digits)}")
            return None

        rstep = warped.shape[0] // 9
        cstep = warped.shape[1] // 9
        board = np.zeros((9, 9), dtype="int")
        for p, tx in digits:
            c = floor(p[0] / cstep)
            r = floor(p[1] / rstep)
            board[r, c] = int(tx) if tx not in 'sS' else 5
        return board

    def try_build_board(self, ori, gray, debug: bool):
        tmp1, tmp2 = ori, gray
        for i in range(4):
            board = self.detect_board(gray, debug)
            if board is not None:
                np.clip(board, 0, 9)
                return board, tmp1, tmp2
            logger.error(f"detect board fail rotate {90 * i}")
            tmp1 = cv2.rotate(tmp1, cv2.ROTATE_90_CLOCKWISE)
            tmp2 = cv2.rotate(tmp2, cv2.ROTATE_90_CLOCKWISE)
        return None, tmp1, tmp2

    def try_solve(self, img, debug):
        logger.debug("will parse image")
        found, puzzle_image, warped = parse_image(img, debug=debug)
        if found:
            puzzle_image, warped = adjust_orientation(self._paddle, puzzle_image, warped)
        return self.process(puzzle_image, warped, debug)

    def process(self, puzzle_image, warped, debug):
        board, puzzle_image, warped = self.try_build_board(puzzle_image, warped, debug)
        if board is None:
            return None

        logger.debug("will call sudoku solver")
        try:
            solution = build_solver(board, debug=debug)
        except Exception as ex:
            logger.error("call solver fail", ex)
            return None

        logger.debug("will draw back")
        puzzle_image = imutils.resize(puzzle_image, width=450)
        cell_locs = self.make_cell_locs(puzzle_image)
        out = draw_back(cell_locs, board, solution, puzzle_image)
        return out

    def try_solve_ndarray(self, image, debug):
        found, puzzle_image, warped = robust_find_puzzle(image, debug)
        if found:
            puzzle_image, warped = adjust_orientation(self._paddle, puzzle_image, warped)
        return self.process(puzzle_image, warped, debug)

    def run(self, img, debug=False):
        res = self.try_solve(img, debug=debug)
        return res

    def run_ndarray(self, image: np.ndarray, debug=False):
        return self.try_solve_ndarray(image, debug)
