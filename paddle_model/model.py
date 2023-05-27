import string
import time

from loguru import logger
from paddleclas import PaddleClas
from paddleocr import PaddleOCR


class PaddleModel:
    def __init__(self):
        self._ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
        self._orient = PaddleClas(model_name="text_image_orientation")

    def predict_orientation(self, img):
        begin = time.time()
        result = self._orient.predict(input_data=img)
        # [{'class_ids': [3, 1], 'scores': [0.83951, 0.10129], 'label_names': ['270', '90']}]
        out = next(result)[0]
        logger.debug(f"orientation result {out}, time cost: {time.time() - begin}")
        scores = out['scores']
        class_ids = out['class_ids']
        confidence, direction = (scores[0], class_ids[0]) if scores[0] > scores[1] else (scores[1], class_ids[1])
        if confidence < 0.7:
            return -1
        return direction

    def predict_cell(self, img):
        begin = time.time()
        out = self._ocr.ocr(img, det=False, cls=False, rec=True)
        logger.debug(f"ocr result {out}, time cost: {time.time() - begin}")
        res = out[0][0][0]
        confidence = out[0][0][1]
        if confidence < 0.4:
            return -2
        val = [ch for ch in res if ch in string.digits + "sS"]
        if len(val) == 1:
            if val[0] in 'sS':
                return 5
            return int(val[0])
        return -1

    def predict_image(self, img):
        out = self._ocr.ocr(img, det=True, rec=True, cls=False)
        for tup in out[0]:
            logger.debug(f"all ocr result {tup}")
        return out[0]
