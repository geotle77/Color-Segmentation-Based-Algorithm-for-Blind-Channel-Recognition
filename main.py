import cv2
import numpy as np
from image_process import ImageProcessor



def color_segmentation(image_processor):
    hsv_image = image_processor.hsv_image
    h, s, v = cv2.split(hsv_image)
    height, width = h.shape
    finish_data = np.zeros_like(h, dtype=np.uint8)

    for i in range(width):  # 注意这里i对应的是宽度（x坐标）
        for j in range(height):  # 注意这里j对应的是高度（y坐标）
            if 190 < h[j, i] < 210 and 60 < s[j, i] < 255 and 60 < v[j, i] < 255:  
                finish_data[j, i] = 255

    return finish_data






if __name__ == '__main__':
    image_processor = ImageProcessor('pics/05.jpg')

    segmented_image = color_segmentation(image_processor)

    cv2.imshow('segmented image', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()