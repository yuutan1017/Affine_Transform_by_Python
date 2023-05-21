
import cv2
import matplotlib.pyplot as plt
import numpy as np


class Affine:

    img = None

    def __init__(self, img):
        self.image = img
        self.h, self.w = self.image.shape[:2]
        self.src = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0]
        ], np.float32)

    # 基本処理
    def warp_affine(self, trans_matrix, magnification=1):
        aff = cv2.getAffineTransform(self.src, trans_matrix)
        return cv2.warpAffine(self.image, aff, (self.w * magnification, self.h * magnification))

    # 恒等変換
    def identity(self):
        return self.warp_affine(self.src)

    # 水平移動
    def shift_x(self, shift_value):
        dst = self.src.copy()
        dst[:, 0] += shift_value
        return self.warp_affine(dst)

    # 垂直移動
    def shift_y(self, shift_value):
        dst = self.src.copy()
        dst[:, 1] += shift_value
        return self.warp_affine(dst)

    # ランダムシフト
    def shift_random(self, shift_values):
        dst = self.src + shift_values.reshape(1, -1).astype(np.float32)
        return self.warp_affine(dst)

    # 拡大・縮小
    def expand(self, ratio, magnification):
        dst = self.src * ratio
        return self.warp_affine(dst, magnification)

    # 水平方向のせん断
    def shear_x(self, shear):
        dst = self.src.copy()
        dst[:, 0] += (shear / self.h * (self.h - self.src[:, 1])).astype(np.float32)
        return self.warp_affine(dst)

    # 垂直方向のせん断
    def shear_y(self, shear):
        dst = self.src.copy()
        dst[:, 1] += (shear / self.w * (self.w - self.src[:, 0])).astype(np.float32)
        return self.warp_affine(dst)

    # 水平反転
    def horizontal_flip(self):
        dst = self.src.copy()
        dst[:, 0] = self.w - self.src[:, 0]
        return self.warp_affine(dst)

    # 垂直反転
    def vertical_flip(self):
        dst = self.src.copy()
        dst[:, 1] = self.h - self.src[:, 1]
        return self.warp_affine(dst)


if __name__ == '__main__':
    image = cv2.imread("Image/cat.jpg")[:, :, ::-1]
    affine = Affine(image)

    # converted = affine.identity()
    # converted = affine.shift_x(150)
    # converted = affine.shift_random(150)
    # converted = affine.expand(1, 2)
    # converted = affine.shear_x(100)
    # converted = affine.shear_y(100)
    # converted = affine.horizontal_flip()
    # converted = affine.vertical_flip()
    converted = 0

    plt.imshow(converted)
    plt.title("AffineTransformation")
    plt.show()
