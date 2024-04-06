import cv2
import numpy as np
from image_process import ImageProcessor
from scipy.ndimage import median_filter as mf, label, sum
from scipy.ndimage.morphology import binary_dilation

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

def median_filter(image, kernel_size=5, area_threshold=100):
    # 对image进行中值滤波
    filtered_image = mf(image, size=kernel_size)

    # 对滤波后的图像进行标记
    labeled_image, num_features = label(filtered_image)

    # 计算每个标记区域的面积
    areas = sum(filtered_image, labeled_image, range(1, num_features+1))

    # 创建一个新的图像，只包含面积大于阈值的区域
    filtered_image = np.zeros_like(filtered_image, dtype=np.uint8)
    for i in range(1, num_features+1):
        if areas[i-1] > area_threshold:
            filtered_image[labeled_image == i] = 255
    return filtered_image

def dilate_image(image, size=15):
    # 创建一个正方形的结构元素
    struct_element = np.ones((size, size))

    # 将图像转换为二值图像
    binary_image = image > 0

    # 对二值图像进行膨胀处理
    dilated_image = binary_dilation(binary_image, structure=struct_element)

    # 将膨胀后的二值图像转换回原来的像素值范围，并转换为uint8类型
    dilated_image = (dilated_image * 255).astype(np.uint8)

    return dilated_image

def label_regions(image):
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

def screen_region(label_image):
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
    

if __name__ == '__main__':
    image_processor = ImageProcessor('pics/01.jpg')

    segmented_image = color_segmentation(image_processor) 
    cv2.imshow('segmented image', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('pics/01_filtered.jpg', segmented_image)
    
    filtered_image = median_filter(segmented_image)
    cv2.imshow('filtered image', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('pics/01_filtered.jpg', filtered_image)

    dilated_image = dilate_image(filtered_image)
    cv2.imshow('dilated image', dilated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('pics/01_dilated.jpg', dilated_image)

    labeled_image = label_regions(dilated_image)
    cv2.imshow('labeled image', labeled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('pics/01_labeled.jpg', labeled_image)

    screen_image = screen_region(labeled_image)
    cv2.imshow('screen image', screen_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('pics/01_screen.jpg', screen_image)