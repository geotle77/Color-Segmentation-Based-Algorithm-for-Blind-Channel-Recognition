import cv2
import numpy as np
import matplotlib.pyplot as plt
import colorsys

class ImageProcessor:
    def __init__(self, path):
        self.image = self.read_image(path)
        self.v = self.cal_v(self.image)
        self.s = self.cal_s(self.image)
        self.h = self.calculate_h(self.image)
        self.hsv_image = self.rgb2hsv(self.image)

    def read_image(self, path):
        return cv2.imread(path)

    def get_pixel_hsv(self, x, y):
        return self.hsv_image[y,x]

    def cal_v(self, image):
        v = np.amax(image, axis=2)
        return v
    

    def cal_s(self, image):
        min_value = np.amin(image, axis=2)
        s = np.zeros_like(self.v)
        mask = (self.v != 0)  # 找到V不等于0的像素
        s[mask] = ((self.v[mask] - min_value[mask]) / self.v[mask]) * 255  # 只在V不等于0的像素上计算S
        return s.astype(np.uint8)

    def calculate_h(self, image):
        min_value = np.amin(image, axis=2)
        diff = self.v - min_value

        r, g, b = cv2.split(image)
        h = np.zeros_like(self.v, dtype=np.int32)

        mask = (self.v == r) & (diff != 0)
        h[mask] = 60 * (g[mask] - b[mask]) / diff[mask]

        mask = (self.v == g) & (diff != 0)
        h[mask] = 120 + 60 * (b[mask] - r[mask]) / diff[mask]

        mask = (self.v == b) & (diff != 0)
        h[mask] = 240 + 60 * (r[mask] - g[mask]) / diff[mask]

        h[h < 0] += 360

        return h

    def hsv_image(self):
        hsv = np.dstack((self.h, self.s, self.v))
        return hsv

    def rgb2hsv(self, image):
        r, g, b = cv2.split(image)
        r, g, b = r / 255.0, g / 255.0, b / 255.0  # 归一化到0~1的范围
        h, s, v = np.vectorize(colorsys.rgb_to_hsv)(r, g, b)  # 使用numpy的广播功能
        h = h * 360
        s = s * 255
        v = v * 255
        hsv = cv2.merge([h, s, v])
        return hsv

    def show_hsv(self):
        h,s,v = cv2.split(self.hsv_image)
        print("Hue values:")
        print(h)

        print("\nSaturation values:")
        print(s)

        print("\nValue values:")
        print(v)




    def display_hsv_components(self):
        plt.imshow(self.h / 360, cmap='hsv')  # 使用HSV色彩映射，imshow只能显示0~255的值，所以改用plt.imshow
        plt.colorbar()  # 显示色彩条
        plt.show()

        cv2.destroyAllWindows()
        cv2.imshow('Saturation', self.s)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('Value', self.v)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def display_image(self):
        cv2.imshow('image', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    image_processor = ImageProcessor('pics/05.jpg')
    image_processor.display_image()
    image_processor.display_hsv_components()
    