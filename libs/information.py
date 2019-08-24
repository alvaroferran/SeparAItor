import cv2
import numpy as np


class Information:

    def __init__(self):
        self.image = np.zeros((256, 256, 3), np.uint8)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.size = 1
        self.color = (0, 0, 0)
        self.thickness = 2
        self.update()

    def update(self, result=None):
        if result is None:
            self.image[:] = (50, 200, 100)
            text = "System ready"
            pos = (20, 150)
            cv2.putText(self.image, text, pos, self.font, self.size,
                        self.color, self.thickness, cv2.LINE_AA)
        else:
            self.image[:] = (200, 100, 50)
            text = "Processing..."
            pos = (10, 50)
            cv2.putText(self.image, text, pos, self.font, self.size,
                        self.color, self.thickness, cv2.LINE_AA)
            text = "Class:"
            pos = (10, 150)
            cv2.putText(self.image, text, pos, self.font, self.size,
                        self.color, self.thickness, cv2.LINE_AA)
            text = result
            pos = (120, 150)
            cv2.putText(self.image, text, pos, self.font, self.size,
                        self.color, self.thickness, cv2.LINE_AA)
