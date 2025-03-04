
import os
import cv2
import numpy as np
from pathlib import Path
from lineless_table_rec import LinelessTableRecognition
from lineless_table_rec.utils_table_recover import format_html, plot_rec_box_with_logic_info, plot_rec_box
from table_cls import TableCls
from wired_table_rec import WiredTableRecognition
# from wired_table_rec.utils import ImageOrientationCorrector
from PIL import Image

class TableStructRec():
    def __init__(self):
        self.lineless_engine = LinelessTableRecognition()
        self.wired_engine = WiredTableRecognition()
        self.table_cls = TableCls()

    def predict(self, image):
        self.lineless_engine = LinelessTableRecognition()
        self.wired_engine = WiredTableRecognition()
        self.table_cls = TableCls()
        # img_orientation_corrector = ImageOrientationCorrector()

        # image.save("save.jpg")
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        # image = img_orientation_corrector(image)


        cls, elasp = self.table_cls(image)
        if cls == 'wired':
            table_engine = self.wired_engine
        else:
            table_engine = self.lineless_engine

        html, elasp, polygons, logic_points, ocr_res = table_engine(image)
        # complete_html = format_html(html)
        return html, None, logic_points, elasp
        # return complete_html, None, logic_points, elasp


    #        return html_code, table_cell_bboxes, logic_points, elapse