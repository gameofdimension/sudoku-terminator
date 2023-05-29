import sys

import cv2
import gradio as gr
from loguru import logger

from lucky.ocr_solver import StructuralSolver


def get_sukoku_solver():
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    solver = StructuralSolver()

    def solve(input_img):
        if input_img is None:
            raise gr.Error("输入错误")
        result = solver.run_ndarray(input_img, debug=False)
        if result is None:
            raise gr.Error("无法识别")
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return solve


demo = gr.Interface(get_sukoku_solver(), gr.Image(tool=None),
                    "image", title="数独终结者", allow_flagging="never")
demo.launch()
