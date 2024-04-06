import cv2
import numpy as np
import os
from image_process import ImageProcessor
from scipy.ndimage import median_filter as mf, label, sum
from scipy.ndimage import binary_dilation

class blind_spot:
    def __init__(self, image_processor,output_path, kernel_size=5, area_threshold=100, dilate_size=15):
        self.image_processor = image_processor
        self.kernel_size = kernel_size
        self.area_threshold = area_threshold
        self.dilate_size = dilate_size
        self.output_path = output_path
        
    def color_segmentation(self):
        hsv_image = self.image_processor.hsv_image
        h, s, v = cv2.split(hsv_image)
        height, width = h.shape
        finish_data = np.zeros_like(h, dtype=np.uint8)

        for i in range(width):  # 注意这里i对应的是宽度（x坐标）
            for j in range(height):  # 注意这里j对应的是高度（y坐标）
                if 190 < h[j, i] < 210 and 60 < s[j, i] < 255 and 60 < v[j, i] < 255:  
                    finish_data[j, i] = 255

        return finish_data

    def median_filter(self, image):
        # 对image进行中值滤波
        filtered_image = mf(image, size=self.kernel_size)

        # 对滤波后的图像进行标记
        labeled_image, num_features = label(filtered_image)

        # 计算每个标记区域的面积
        areas = sum(filtered_image, labeled_image, range(1, num_features+1))

        # 创建一个新的图像，只包含面积大于阈值的区域
        filtered_image = np.zeros_like(filtered_image, dtype=np.uint8)
        for i in range(1, num_features+1):
            if areas[i-1] > self.area_threshold:
                filtered_image[labeled_image == i] = 255
        return filtered_image

    def dilate_image(self,image):
        # 创建一个正方形的结构元素
        struct_element = np.ones((self.dilate_size,self.dilate_size))

        # 将图像转换为二值图像
        binary_image = image > 0

        # 对二值图像进行膨胀处理
        dilated_image = binary_dilation(binary_image, structure=struct_element)

        # 将膨胀后的二值图像转换回原来的像素值范围，并转换为uint8类型
        dilated_image = (dilated_image * 255).astype(np.uint8)

        return dilated_image

    def label_regions(self,image):
        height, width = image.shape
        class_num = 0
        
        stack = []
        for i in range(height):
            for j in range(width):
                if image[i, j] == 255:
                    stack.append((i, j))
                    class_num += 1
                    while stack:
                        x0, y0 = stack.pop()
                        if 0 <= x0 < height and 0 <= y0 < width and image[x0, y0] == 255:
                            image[x0, y0] = class_num
                            stack.extend([(x0-1, y0), (x0+1, y0), (x0, y0-1), (x0, y0+1),
                                        (x0-1, y0-1), (x0-1, y0+1), (x0+1, y0-1), (x0+1, y0+1)])
        for i in range(height):
            for j in range(width):
                if image[i, j] != 0:
                    image[i, j] = 255 * image[i, j] / class_num
        return image

    def screen_region(self,label_image):
        # 对图像进行标记
        labeled_image, num_features = label(label_image)

        # 计算每个标记区域的面积
        areas = sum(label_image, labeled_image, range(1, num_features+1))

        # 找到面积最大的区域的标签
        largest_label = np.argmax(areas) + 1

        # 创建一个新的图像，只包含面积最大的区域
        new_image = np.zeros_like(label_image, dtype=np.uint8)
        new_image[labeled_image == largest_label] = 255

        return new_image
    
    def run_show(self):
        segmented_image = self.color_segmentation()
        self.show_image(segmented_image)
        filtered_image = self.median_filter(segmented_image)
        self.show_image(filtered_image)
        dilated_image = self.dilate_image(filtered_image)
        self.show_image(dilated_image)
        labeled_image = self.label_regions(dilated_image)
        self.show_image(labeled_image)
        screen_image = self.screen_region(labeled_image)
        self.show_image(screen_image)
        self.save_image(screen_image, 'output.jpg')

    def run(self):
        segmented_image = self.color_segmentation()
        filtered_image = self.median_filter(segmented_image)
        dilated_image = self.dilate_image(filtered_image)
        labeled_image = self.label_regions(dilated_image)
        screen_image = self.screen_region(labeled_image)
        self.save_image(screen_image, 'output.jpg')

    def show_image(self, image):
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_image(self, image, name):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        cv2.imwrite(os.path.join(self.output_path, name), image)



if __name__ == '__main__':
    image_processor = ImageProcessor('pics/01.jpg')

    bs = blind_spot(image_processor,'./pics/output', kernel_size=5, area_threshold=100, dilate_size=15)
    bs.run_show()