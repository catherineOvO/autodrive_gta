from mss import mss

from PIL import Image

from time import time


class ScreenShoter(object):
    def __init__(self):
        self.sct = mss()
        self.monitor = self.sct.monitors[1]

    def shot(self):
        sct_img = self.sct.grab(self.monitor)
        # Convert to PIL/Pillow Image
        return Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')


if __name__ == '__main__':
    start = time()
    shoter = ScreenShoter()
    for i in range(1000):
        img = shoter.shot()
    stop = time()

    print("%.3f", stop - start)
