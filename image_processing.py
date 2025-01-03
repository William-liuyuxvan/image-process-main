"""
图像处理系统的核心处理模块
创建时间：2024-12-27
作者：作者1、作者2
功能：提供图像处理的各种基础功能，包括：
- 图像的加载和保存
- 几何变换（缩放、平移、旋转、翻转、剪切）
- 像素变换（图像合成、灰度变换、直方图处理）
- 图像去噪（中值滤波、均值滤波、高斯滤波、频域滤波）
- 图像锐化（拉普拉斯锐化、USM锐化）
- 边缘检测（Sobel、Canny、Prewitt）
- 图像分割（阈值分割、区域生长、区域分裂合并）
- 深度学习图像分类
"""

# 导入必要的库
import numpy as np  # 用于数值计算和数组操作
import cv2  # OpenCV库,用于图像处理
from PIL import Image  # Python图像处理库
import os  # 操作系统接口

class ImageProcessor:
    """
    图像处理类，提供各种图像处理功能
    创建时间：2024-12-27
    """
    
    def __init__(self):
        """
        初始化图像处理器
        创建时间：2024-12-27
        作者：作者1
        功能：初始化图像处理器的基本属性
        """
        self.image = None  # 存储当前处理的图像
        
    def load_image(self, path):
        """
        加载图像文件
        创建时间：2024-12-27
        作者：作者1
        功能：从指定路径加载图像文件到内存
        
        参数：
            path (str): 图像文件的路径
            
        返回：
            ndarray: 加载的图像数组，如果加载失败返回None
            
        注意：
            使用OpenCV的imread函数加载图像，格式为BGR
        """
        self.image = cv2.imread(path)
        return self.image
        
    def save_image(self, path):
        """
        保存图像到文件
        创建时间：2024-12-27
        作者：作者1
        功能：将当前处理的图像保存到指定路径
        
        参数：
            path (str): 保存图像的目标路径
            
        注意：
            确保目标路径所在的目录存在且有写入权限
        """
        if self.image is not None:
            cv2.imwrite(path, self.image)
            
    # 几何变换功能
    def scale_image(self, scale_x, scale_y):
        """图像缩放
        创建时间：2024-12-27
        作者：作者1
        Args:
            scale_x: x方向的缩放比例
            scale_y: y方向的缩放比例
        Returns:
            缩放后的图像数组
        """
        if self.image is not None:
            try:
                # 获取原始图像尺寸
                height, width = self.image.shape[:2]
                print(f"原始尺寸: {width}x{height}")
                
                # 计算新尺寸
                new_width = max(1, int(width * float(scale_x)))
                new_height = max(1, int(height * float(scale_y)))
                print(f"目标尺寸: {new_width}x{new_height}")
                
                # 创建新图像
                result = np.zeros((new_height, new_width, 3), dtype=np.uint8)
                
                # 计算映射关系
                for y in range(new_height):
                    for x in range(new_width):
                        # 计算原图对应位置
                        orig_x = min(int(x / scale_x), width - 1)
                        orig_y = min(int(y / scale_y), height - 1)
                        
                        # 复制像素值
                        result[y, x] = self.image[orig_y, orig_x]
                
                # 更新当前图像
                self.image = result
                return result
                
            except Exception as e:
                print(f"缩放出错: {str(e)}")
                return self.image
        return None
            
    def translate_image(self, dx, dy):
        """图像平移
        创建时间：2024-12-27
        作者：作者1
        Args:
            dx: x方向的平移量
            dy: y方向的平移量
        Returns:
            平移后的图像数组
        """
        if self.image is not None:
            height, width = self.image.shape[:2]
            translated = np.zeros_like(self.image)
            
            for i in range(height):
                for j in range(width):
                    #计算每个像素点的新坐标
                    new_x = j + dx
                    new_y = i + dy
                    #如果新坐标在目标图像边界内，则将原像素点的值赋给目标位置。
                    if 0 <= new_x < width and 0 <= new_y < height:
                        translated[new_y, new_x] = self.image[i, j]
            return translated
        return None
            
    def rotate_image(self, angle):
        """图像旋转
        创建时间：2024-12-27
        作者：作者1
        Args:
            angle: 旋转角度(度)
        Returns:
            旋转后的图像数组
        """
        if self.image is not None:
            # 将角度转换为弧度
            radian = np.deg2rad(angle)
            height, width = self.image.shape[:2]
            
            # 计算旋转后的图像大小
            cos_theta = np.abs(np.cos(radian))
            sin_theta = np.abs(np.sin(radian))
            new_width = int(width * cos_theta + height * sin_theta)
            new_height = int(height * cos_theta + width * sin_theta)
            
            # 创建新图像
            rotated = np.zeros((new_height, new_width, self.image.shape[2]), dtype=np.uint8)
            
            # 计算中心点
            center_x = width // 2
            center_y = height // 2
            new_center_x = new_width // 2
            new_center_y = new_height // 2
            
            # 对每个像素进行旋转变换
            for i in range(new_height):
                for j in range(new_width):
                    # 将坐标原点移到中心
                    x = j - new_center_x
                    y = i - new_center_y
                    
                    # 逆旋转变换
                    old_x = int(x * np.cos(radian) + y * np.sin(radian) + center_x)
                    old_y = int(-x * np.sin(radian) + y * np.cos(radian) + center_y)
                    
                    # 检查坐标是否在原图范围内
                    if 0 <= old_x < width and 0 <= old_y < height:
                        rotated[i, j] = self.image[old_y, old_x]
            
            return rotated
            
    def flip_image(self, flip_code):
        """图像翻转
        创建时间：2024-12-27
        作者：作者1
        Args:
            flip_code: 翻转方式
                0-上下翻转
                1-左右翻转
                -1-上下左右翻转
        Returns:
            翻转后的图像数组
        """
        if self.image is not None:
            try:
                height, width = self.image.shape[:2]
                flipped = np.zeros_like(self.image)
                
                for i in range(height):
                    for j in range(width):
                        if flip_code == 0:  # 上下翻转
                            flipped[height-1-i,j] = self.image[i,j]
                        elif flip_code == 1:  # 左右翻转
                            flipped[i,width-1-j] = self.image[i,j]
                        else:  # 上下左右翻转
                            flipped[height-1-i,width-1-j] = self.image[i,j]
                return flipped
                
            except Exception as e:
                print(f"翻转失败: {str(e)}")
                return self.image
        return None

    def image_addition(self, image2):
        """图像加法运算
        创建时间：2024-12-27
        作者：作者1
        Args:
            image2: 第二张图像
        Returns:
            两图像相加的结果数组
        """
        if self.image is not None and image2 is not None:
            # 调整第二张图片的大小以匹配第一张图片
            if self.image.shape != image2.shape:
                image2 = cv2.resize(image2, (self.image.shape[1], self.image.shape[0]))
            
            result = np.zeros_like(self.image)
            for i in range(self.image.shape[0]):
                for j in range(self.image.shape[1]):
                    for k in range(self.image.shape[2]):
                        # 饱和加法
                        pixel_sum = int(self.image[i,j,k]) + int(image2[i,j,k])
                        result[i,j,k] = min(255, pixel_sum)
            return result
        return None
            
    def image_subtraction(self, image2):
        """图像减法运算
        创建时间：2024-12-27
        作者：作者1
        Args:
            image2: 第二张图像
        Returns:
            两图像相减的结果数组
        """
        if self.image is not None and image2 is not None:
            # 调整第二张图片的大小以匹配第一张图片
            if self.image.shape != image2.shape:
                image2 = cv2.resize(image2, (self.image.shape[1], self.image.shape[0]))
            
            result = np.zeros_like(self.image)
            for i in range(self.image.shape[0]):
                for j in range(self.image.shape[1]):
                    for k in range(self.image.shape[2]):
                        # 饱和减法
                        pixel_diff = int(self.image[i,j,k]) - int(image2[i,j,k])
                        result[i,j,k] = max(0, pixel_diff)
            return result
        return None
            
    def image_and(self, image2):
        """图像与运算
        创建时间：2024-12-27
        作者：作者1
        Args:
            image2: 第张图像
        Returns:
            两图像按位与的结果数组
        """
        if self.image is not None and image2 is not None:
            if self.image.shape != image2.shape:
                image2 = cv2.resize(image2, (self.image.shape[1], self.image.shape[0]))
            result = np.zeros_like(self.image)
            for i in range(self.image.shape[0]):
                for j in range(self.image.shape[1]):
                    for k in range(self.image.shape[2]):
                        result[i,j,k] = self.image[i,j,k] & image2[i,j,k]
            return result
        return None
            
    def image_or(self, image2):
        """图像或运算
        创建时间：2024-12-27
        作者：作者1
        Args:
            image2: 第二张图像
        Returns:
            两图像按位或的结果数组
        """
        if self.image is not None and image2 is not None:
            if self.image.shape != image2.shape:
                image2 = cv2.resize(image2, (self.image.shape[1], self.image.shape[0]))
            result = np.zeros_like(self.image)
            for i in range(self.image.shape[0]):
                for j in range(self.image.shape[1]):
                    for k in range(self.image.shape[2]):
                        result[i,j,k] = self.image[i,j,k] | image2[i,j,k]
            return result
        return None
            
    def image_not(self):
        """图像补运算
        创建时间：2024-12-27
        作者：作者1
        Returns:
            图像取反的结果数组
        """
        if self.image is not None:
            result = np.zeros_like(self.image)
            for i in range(self.image.shape[0]):
                for j in range(self.image.shape[1]):
                    for k in range(self.image.shape[2]):
                        result[i,j,k] = 255 - self.image[i,j,k]
            return result
            
    def image_xor(self, image2):
        """图像异或运算
        创建时间：2024-12-27
        作者：作者1
        Args:
            image2: 第二张图像
        Returns:
            两图像按位异或的结果数组
        """
        if self.image is not None and image2 is not None:
            if self.image.shape != image2.shape:
                image2 = cv2.resize(image2, (self.image.shape[1], self.image.shape[0]))
            result = np.zeros_like(self.image)
            for i in range(self.image.shape[0]):
                for j in range(self.image.shape[1]):
                    for k in range(self.image.shape[2]):
                        result[i,j,k] = self.image[i,j,k] ^ image2[i,j,k]
            return result
        return None
            
    def to_grayscale(self):
        """转换为灰度图像
        创建时间：2024-12-27
        作者：作者1
        使用加权平均法:Y = 0.299R + 0.587G + 0.114B
        Returns:
            灰度图像数组
        """
        if self.image is not None:
            # 使用加权平均法
            return np.uint8(self.image[:,:,0] * 0.114 + 
                          self.image[:,:,1] * 0.587 + 
                          self.image[:,:,2] * 0.299)
            
    def adjust_brightness_contrast(self, brightness=0, contrast=1.0):
        """调整亮度和对比度
        创建时间：2024-12-27
        作者：作者1
        Args:
            brightness: 亮度调整值，正值增加亮度，负值降低亮度，范围[-255,255]
            contrast: 对比度调整值，>1增加对比度，<1降低对比度，范围[0,3]
        Returns:
            调整后的图像数组
        """
        if self.image is not None:
            try:
                # 检查图像是灰度图还是彩色图
                is_gray = len(self.image.shape) == 2
                
                # 限制参数范围
                brightness = max(-255, min(255, brightness))
                contrast = max(0, min(3, contrast))
                
                # 将图像转换为浮点型进行计算
                if is_gray:
                    height, width = self.image.shape
                    result = np.zeros((height, width), dtype=np.float32)
                    
                    # 先调整对比度，再调整亮度
                    for i in range(height):
                        for j in range(width):
                            # 归一化到[-1,1]范围
                            pixel = (float(self.image[i,j]) - 128) / 128
                            # 应用对比度
                            pixel = pixel * contrast
                            # 还原到[0,255]范围并调整亮度
                            pixel = (pixel * 128 + 128) + brightness
                            # 裁剪到有效范围
                            result[i,j] = np.clip(pixel, 0, 255)
                            
                else:
                    height, width = self.image.shape[:2]
                    result = np.zeros_like(self.image, dtype=np.float32)
                    
                    # 对每个通道分别处理
                    for c in range(3):
                        for i in range(height):
                            for j in range(width):
                                # 归一化到[-1,1]范围
                                pixel = (float(self.image[i,j,c]) - 128) / 128
                                # 应用对比度
                                pixel = pixel * contrast
                                # 还原到[0,255]范围并调整亮度
                                pixel = (pixel * 128 + 128) + brightness
                                # 裁剪到有效范围
                                result[i,j,c] = np.clip(pixel, 0, 255)
                
                return result.astype(np.uint8)
                
            except Exception as e:
                print(f"亮度对比度调整失败: {str(e)}")
                return self.image
        return None
            
    def calculate_histogram(self):
        """计算图像直方图
        创建时间：2024-12-27
        作者：作者1
        Returns:
            包含BGR三个通道直方图的列表
        """
        if self.image is not None:
            hist = []
            for channel in range(3):  # BGR通道
                h = np.zeros(256)
                channel_data = self.image[:,:,channel].ravel()#提取当前通道的二维像素数据，并将其展为一维数组
                #遍历当前通道的所有像素值 pixel，将其作为索引，累加到数组 h 中，完成频数统计。
                for pixel in channel_data:
                    h[pixel] += 1
                hist.append(h)
            return hist
            
    def equalize_histogram(self):
        """直方图均衡化
        创建时间：2024-12-27
        作者：作者1
        Returns:
            均衡化后的图像数组
        """
        if self.image is not None:
            # 转换到YUV空间
            yuv = cv2.cvtColor(self.image, cv2.COLOR_BGR2YUV)
            y = yuv[:,:,0]
            
            # 计算累积分布函数
            hist = np.zeros(256)
            for pixel in y.ravel():
                hist[pixel] += 1
            cdf = hist.cumsum()
            cdf_normalized = cdf * 255 / cdf[-1]
            
            # 均衡化
            yuv[:,:,0] = cdf_normalized[y]
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    def median_filter(self, kernel_size=5):
        """自定义中值滤波
        创建时间：2024-12-27
        作者：作者1
        Args:
            kernel_size: 滤波核大小，默认为5
        Returns:
            滤波后的图像数组
        """
        if self.image is not None:
            try:
                # 确保核大小是奇数
                kernel_size = max(5, kernel_size)
                if kernel_size % 2 == 0:
                    kernel_size += 1

                # 检查图像是灰度图还是彩色图
                is_gray = len(self.image.shape) == 2
                
                # 创建输出图像
                if is_gray:
                    height, width = self.image.shape
                    result = np.zeros((height, width), dtype=np.uint8)
                else:
                    height, width = self.image.shape[:2]
                    result = np.zeros_like(self.image)

                # 计算填充大小
                pad = kernel_size // 2

                if is_gray:
                    # 处理灰度图
                    padded = np.pad(self.image, ((pad, pad), (pad, pad)), mode='edge')
                    for i in range(height):
                        for j in range(width):
                            window = padded[i:i + kernel_size, j:j + kernel_size].flatten()
                            window_sorted = sorted(window)
                            mid_index = len(window_sorted) // 2
                            result[i, j] = window_sorted[mid_index]
                else:
                    # 处理彩色图
                    for c in range(3):  # BGR三个通道
                        channel = self.image[:, :, c]
                        padded = np.pad(channel, ((pad, pad), (pad, pad)), mode='edge')
                        for i in range(height):
                            for j in range(width):
                                window = padded[i:i + kernel_size, j:j + kernel_size].flatten()
                                window_sorted = sorted(window)
                                mid_index = len(window_sorted) // 2
                                result[i, j, c] = window_sorted[mid_index]

                return result

            except Exception as e:
                print(f"中值滤波失败: {str(e)}")
                return self.image
        return None
            
    def mean_filter(self, kernel_size=5):
        """均值滤波
        创建时间：2024-12-27
        作者：作者1
        Args:
            kernel_size: 滤波核大小，默认为5
        Returns:
            滤波后的图像数组
        """
        if self.image is not None:
            try:
                # 确保核大小是奇数
                kernel_size = max(5, kernel_size)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                # 生成均值滤波核
                kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
                
                # 检查图像是灰度图还是彩色图
                is_gray = len(self.image.shape) == 2
                
                # 获取图像尺寸
                if is_gray:
                    height, width = self.image.shape
                    result = np.zeros((height, width), dtype=np.uint8)
                    # 边缘填充
                    pad = kernel_size // 2
                    padded = np.pad(self.image, ((pad, pad), (pad, pad)), mode='edge')
                    
                    # 应用滤波器
                    for i in range(height):
                        for j in range(width):
                            window = padded[i:i+kernel_size, j:j+kernel_size]
                            result[i,j] = np.sum(window * kernel)
                else:
                    height, width = self.image.shape[:2]
                    result = np.zeros_like(self.image)
                    pad = kernel_size // 2
                    
                    # 对每个通道分别处理
                    for c in range(3):
                        padded = np.pad(self.image[:,:,c], ((pad, pad), (pad, pad)), mode='edge')
                        for i in range(height):
                            for j in range(width):
                                window = padded[i:i+kernel_size, j:j+kernel_size]
                                result[i,j,c] = np.sum(window * kernel)
                
                return result.astype(np.uint8)
                
            except Exception as e:
                print(f"均值滤波失败: {str(e)}")
                return self.image
        return None
            
    def gaussian_filter(self, kernel_size=5, sigma=1.0):
        """高斯滤波
        创建时间：2024-12-27
        作者：作者1
        Args:
            kernel_size: 滤波核大小，默认为5
            sigma: 高斯函数的标准差，默认为1.0
        Returns:
            滤波后的图像数组
        """
        if self.image is not None:
            try:
                # 确保核大小是奇数
                kernel_size = max(3, kernel_size)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                # 生成高斯核
                center = kernel_size // 2
                kernel = np.zeros((kernel_size, kernel_size))
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        x, y = i - center, j - center
                        kernel[i, j] = np.exp(-(x**2 + y**2)/(2*sigma**2))
                
                # 归一化核
                kernel = kernel / kernel.sum()
                
                # 检查图像是灰度图还是彩色图
                is_gray = len(self.image.shape) == 2
                
                # 获取图像尺寸
                if is_gray:
                    height, width = self.image.shape
                    result = np.zeros((height, width), dtype=np.uint8)
                    # 边缘填充
                    pad = kernel_size // 2
                    padded = np.pad(self.image, ((pad, pad), (pad, pad)), mode='edge')
                    
                    # 应用滤波器
                    for i in range(height):
                        for j in range(width):
                            window = padded[i:i+kernel_size, j:j+kernel_size]
                            result[i,j] = np.sum(window * kernel)
                else:
                    height, width = self.image.shape[:2]
                    result = np.zeros_like(self.image)
                    pad = kernel_size // 2
                    
                    # 对每个通道分别处理
                    for c in range(3):
                        padded = np.pad(self.image[:,:,c], ((pad, pad), (pad, pad)), mode='edge')
                        for i in range(height):
                            for j in range(width):
                                window = padded[i:i+kernel_size, j:j+kernel_size]
                                result[i,j,c] = np.sum(window * kernel)
                
                return result.astype(np.uint8)
                
            except Exception as e:
                print(f"高斯滤波失败: {str(e)}")
                return self.image
        return None
            
    def frequency_filter(self, d0, filter_type='lowpass'):
        """频域滤波
        创建时间：2024-12-27
        作者：作者1
        Args:
            d0: 截止频率
            filter_type: 滤波类型，'lowpass'或'highpass'
        Returns:
            滤波后的图像数组
        """
        if self.image is not None:
            try:
                # 转换为灰度图像
                if len(self.image.shape) > 2:
                    gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.image.copy()
                
                rows, cols = gray.shape
                
                # 确保尺寸是2的幂次方
                nrows = 2 ** int(np.ceil(np.log2(rows)))
                ncols = 2 ** int(np.ceil(np.log2(cols)))
                padded = np.pad(gray, ((0, nrows-rows), (0, ncols-cols)), mode='constant')
                
                # 快速傅里叶变换实现
                def custom_fft(x):
                    x = np.asarray(x, dtype=np.complex128)
                    N = x.shape[0]
                    
                    # 确保长度是2的幂次方
                    if N & (N-1) != 0:
                        raise ValueError("FFT长度必须是2的幂次方")
                    elif N <= 1:
                        return x
                    
                    even = custom_fft(x[::2])
                    odd = custom_fft(x[1::2])
                    factor = np.exp(-2j * np.pi * np.arange(N//2) / N)
                    return np.concatenate([even + factor * odd,
                                        even - factor * odd])
                
                # 快速傅里叶逆变换实现
                def custom_ifft(x):
                    x = np.asarray(x, dtype=np.complex128)
                    N = x.shape[0]
                    
                    if N & (N-1) != 0:
                        raise ValueError("IFFT长度必须是2的幂次方")
                    elif N <= 1:
                        return x
                    
                    even = custom_ifft(x[::2])
                    odd = custom_ifft(x[1::2])
                    factor = np.exp(2j * np.pi * np.arange(N//2) / N)
                    return (np.concatenate([even + factor * odd,
                                         even - factor * odd])) / 2
                
                # 对图像进行预处理
                padded = padded.astype(np.float32)
                for i in range(nrows):
                    for j in range(ncols):
                        padded[i,j] = padded[i,j] * (-1)**(i+j)
                
                # 对每行进行快速傅里叶变换
                temp = np.zeros((nrows, ncols), dtype=np.complex128)
                for i in range(nrows):
                    temp[i,:] = custom_fft(padded[i,:])
                
                # 对每列进行快速傅里叶变换
                dft = np.zeros((nrows, ncols), dtype=np.complex128)
                for j in range(ncols):
                    dft[:,j] = custom_fft(temp[:,j])
                
                # 创建滤波器
                mask = np.zeros((nrows, ncols))
                center_x, center_y = nrows//2, ncols//2
                for i in range(nrows):
                    for j in range(ncols):
                        d = np.sqrt((i-center_x)**2 + (j-center_y)**2)
                        if filter_type == 'lowpass':
                            mask[i,j] = 1 if d <= d0 else 0
                        else:  # highpass
                            mask[i,j] = 0 if d <= d0 else 1
                
                # 应用滤波器
                filtered = dft * mask
                
                # 对每行进行快速傅里叶逆变换
                temp = np.zeros((nrows, ncols), dtype=np.complex128)
                for i in range(nrows):
                    temp[i,:] = custom_ifft(filtered[i,:])
                
                # 对每列进行快速傅里叶逆变换
                result = np.zeros((nrows, ncols), dtype=np.complex128)
                for j in range(ncols):
                    result[:,j] = custom_ifft(temp[:,j])
                
                # 后处理
                for i in range(nrows):
                    for j in range(ncols):
                        result[i,j] = result[i,j] * (-1)**(i+j)
                
                # 取实部并裁剪回原始大小
                result = np.real(result[:rows, :cols])
                
                # 归一化到0-255范围
                result = ((result - result.min()) * 255 / (result.max() - result.min())).astype(np.uint8)
                
                # 如果原图是彩色图，转回彩色
                if len(self.image.shape) > 2:
                    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                
                return result
                
            except Exception as e:
                print(f"频域滤波失败: {str(e)}")
                return self.image
        return None

    def laplacian_sharpen(self):
        """拉普拉斯锐化
        创建时间：2024-12-30
        作者：作者2
        Returns:
            锐化后的图像数组
        """
        if self.image is not None:
            try:
                # 检查图像是灰度图还是彩色图
                is_gray = len(self.image.shape) == 2
                
                # 定义拉普拉斯算子
                kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])
                
                if is_gray:
                    # 处理灰度图像
                    height, width = self.image.shape
                    result = np.zeros((height, width), dtype=np.float32)
                    padded = np.pad(self.image, ((1, 1), (1, 1)), mode='edge')
                    
                    # 应用拉普拉斯算子
                    for i in range(height):
                        for j in range(width):
                            window = padded[i:i+3, j:j+3]
                            laplacian = np.sum(window * kernel)
                            # 锐化 = 原图 - 拉普拉斯
                            result[i,j] = self.image[i,j] - laplacian
                            
                else:
                    # 处理彩色图像
                    height, width = self.image.shape[:2]
                    result = np.zeros_like(self.image, dtype=np.float32)
                    
                    # 对每个通道分别处理
                    for c in range(3):
                        padded = np.pad(self.image[:,:,c], ((1, 1), (1, 1)), mode='edge')
                        for i in range(height):
                            for j in range(width):
                                window = padded[i:i+3, j:j+3]
                                laplacian = np.sum(window * kernel)
                                # 锐化 = 原图 - 拉普拉斯
                                result[i,j,c] = self.image[i,j,c] - laplacian
                
                # 裁剪到0-255范围
                result = np.clip(result, 0, 255).astype(np.uint8)
                return result
                
            except Exception as e:
                print(f"拉普拉斯锐化失败: {str(e)}")
                return self.image
        return None
            
    def unsharp_masking(self, amount=1.5, radius=5, threshold=10):
        """非线性锐化(USM)
        创建时间：2024-12-30
        作者：作者2
        Args:
            amount: 锐化强度，默认1.5
            radius: 高斯模糊半径，默认5
            threshold: 边缘阈值，默认10
        Returns:
            锐化后的图像数组
        """
        if self.image is not None:
            try:
                # 检查图像是灰度图还是彩色图
                is_gray = len(self.image.shape) == 2
                
                # 1. 对原图进行高斯模糊
                blurred = self.gaussian_filter(radius, sigma=radius/3)
                
                if is_gray:
                    # 处理灰度图像
                    height, width = self.image.shape
                    result = np.zeros((height, width), dtype=np.float32)
                    
                    # 2. 计算非锐化掩模
                    for i in range(height):
                        for j in range(width):
                            # 计算原图与模糊图的差值
                            diff = float(self.image[i,j]) - float(blurred[i,j])
                            # 只对超过阈值的差值进行增强
                            if abs(diff) > threshold:
                                result[i,j] = self.image[i,j] + amount * diff
                            else:
                                result[i,j] = self.image[i,j]
                else:
                    # 处理彩色图像
                    height, width = self.image.shape[:2]
                    result = np.zeros_like(self.image, dtype=np.float32)
                    
                    # 对每个通道分别处理
                    for c in range(3):
                        for i in range(height):
                            for j in range(width):
                                # 计算原图与模糊图的差值
                                diff = float(self.image[i,j,c]) - float(blurred[i,j,c])
                                # 只对超过阈值的差值进行增强
                                if abs(diff) > threshold:
                                    result[i,j,c] = self.image[i,j,c] + amount * diff
                                else:
                                    result[i,j,c] = self.image[i,j,c]
                
                # 裁剪到0-255范围
                result = np.clip(result, 0, 255).astype(np.uint8)
                return result
                
            except Exception as e:
                print(f"USM锐化失败: {str(e)}")
                return self.image
        return None
            
    def sobel_edge(self):
        """Sobel边缘检测
        创建时间：2024-12-30
        作者：作者2
        Returns:
            边缘检测后的图像数组
        """
        if self.image is not None:
            try:
                # 转换为灰度图像
                if len(self.image.shape) > 2:
                    gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.image.copy()
                
                # 定义Sobel算子
                kernel_x = np.array([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]])
                
                kernel_y = np.array([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]])
                
                # 获取图像尺寸
                height, width = gray.shape
                result = np.zeros((height, width), dtype=np.uint8)
                
                # 边缘填充
                padded = np.pad(gray, ((1, 1), (1, 1)), mode='edge')
                
                # 计算梯度
                for i in range(height):
                    for j in range(width):
                        # 提取3x3窗口
                        window = padded[i:i+3, j:j+3]
                        # 计算x和y方向的梯度
                        gx = np.sum(window * kernel_x)
                        gy = np.sum(window * kernel_y)
                        # 计算梯度幅值
                        grad = np.sqrt(gx*gx + gy*gy)
                        result[i,j] = np.clip(grad, 0, 255)
                
                return result
                
            except Exception as e:
                print(f"Sobel边缘检测失败: {str(e)}")
                return self.image
        return None
            
    def canny_edge(self, threshold1=100, threshold2=200):
        """Canny边缘检测
        创建时间：2024-12-30
        作者：作者2
        Args:
            threshold1: 低阈值
            threshold2: 高阈值
        Returns:
            边缘检测后的图像数组
        """
        if self.image is not None:
            try:
                # 转换为灰度图像
                if len(self.image.shape) > 2:
                    gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.image.copy()
                
                # 1. 高斯平滑
                kernel_size = 5
                sigma = 1.4
                smoothed = self.gaussian_filter(kernel_size, sigma)
                
                # 2. 计算梯度
                kernel_x = np.array([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]])
                
                kernel_y = np.array([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]])
                
                height, width = gray.shape
                padded = np.pad(gray, ((1, 1), (1, 1)), mode='edge')
                
                # 初始化梯度和方向数组
                magnitude = np.zeros((height, width))
                direction = np.zeros((height, width))
                
                # 计算梯度和方向
                for i in range(height):
                    for j in range(width):
                        window = padded[i:i+3, j:j+3]
                        gx = np.sum(window * kernel_x)
                        gy = np.sum(window * kernel_y)
                        magnitude[i,j] = np.sqrt(gx*gx + gy*gy)
                        direction[i,j] = np.arctan2(gy, gx) * 180 / np.pi
                
                # 3. 非极大值抑制
                suppressed = np.zeros((height, width))
                for i in range(1, height-1):
                    for j in range(1, width-1):
                        angle = direction[i,j]
                        # 将角度转换到0-180度
                        if angle < 0:
                            angle += 180
                            
                        # 根据梯度方向选择比较的像素
                        if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                            prev = magnitude[i,j-1]
                            next = magnitude[i,j+1]
                        elif 22.5 <= angle < 67.5:
                            prev = magnitude[i+1,j-1]
                            next = magnitude[i-1,j+1]
                        elif 67.5 <= angle < 112.5:
                            prev = magnitude[i+1,j]
                            next = magnitude[i-1,j]
                        else:
                            prev = magnitude[i-1,j-1]
                            next = magnitude[i+1,j+1]
                            
                        # 如果当前点是局部最大值，保留
                        if magnitude[i,j] >= prev and magnitude[i,j] >= next:
                            suppressed[i,j] = magnitude[i,j]
                
                # 4. 双阈值处理和连接
                result = np.zeros((height, width), dtype=np.uint8)
                strong = 255
                weak = 50
                
                strong_i, strong_j = np.where(suppressed >= threshold2)
                weak_i, weak_j = np.where((suppressed >= threshold1) & (suppressed < threshold2))
                
                result[strong_i, strong_j] = strong
                result[weak_i, weak_j] = weak
                
                # 连接边缘
                for i in range(1, height-1):
                    for j in range(1, width-1):
                        if result[i,j] == weak:
                            if np.any(result[i-1:i+2, j-1:j+2] == strong):
                                result[i,j] = strong
                            else:
                                result[i,j] = 0
                
                return result
                
            except Exception as e:
                print(f"Canny边缘检测失败: {str(e)}")
                return self.image
        return None
            
    def prewitt_edge(self):
        """Prewitt边缘检测
        创建时间：2024-12-30
        作者：作者2
        Returns:
            边缘检测后的图像数组
        """
        if self.image is not None:
            try:
                # 转换为灰度图像
                if len(self.image.shape) > 2:
                    gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.image.copy()
                
                # 定义Prewitt算子
                kernel_x = np.array([[1, 1, 1],
                                   [0, 0, 0],
                                   [-1, -1, -1]])
                
                kernel_y = np.array([[-1, 0, 1],
                                   [-1, 0, 1],
                                   [-1, 0, 1]])
                
                # 获取图像尺寸
                height, width = gray.shape
                result = np.zeros((height, width), dtype=np.uint8)
                
                # 边缘填充
                padded = np.pad(gray, ((1, 1), (1, 1)), mode='edge')
                
                # 计算梯度
                for i in range(height):
                    for j in range(width):
                        # 提取3x3窗口
                        window = padded[i:i+3, j:j+3]
                        # 计算x和y方向的梯度
                        gx = np.sum(window * kernel_x)
                        gy = np.sum(window * kernel_y)
                        # 计算梯度幅值
                        grad = np.sqrt(gx*gx + gy*gy)
                        result[i,j] = np.clip(grad, 0, 255)
                
                return result
                
            except Exception as e:
                print(f"Prewitt边缘检测失败: {str(e)}")
                return self.image
        return None

    def threshold_segmentation(self, threshold, max_val=255):
        """阈值分割
        创建时间：2024-12-30
        作者：作者2
        Args:
            threshold: 阈值
            max_val: 最大值，默认255
        Returns:
            分割后的二值图像数组
        """
        if self.image is not None:
            try:
                # 转换为灰度图像
                if len(self.image.shape) > 2:
                    gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.image.copy()
                
                # 获取图像尺寸
                height, width = gray.shape
                result = np.zeros((height, width), dtype=np.uint8)
                
                # 应用阈值
                for i in range(height):
                    for j in range(width):
                        result[i,j] = max_val if gray[i,j] > threshold else 0
                
                return result
                
            except Exception as e:
                print(f"阈值分割失败: {str(e)}")
                return self.image
        return None

    def otsu_threshold(self):
        """Otsu自适应阈值分割
        创建时间：2024-12-30
        作者：作者2
        Returns:
            分割后的二值图像数组
        """
        if self.image is not None:
            try:
                # 转换为灰度图像
                if len(self.image.shape) > 2:
                    gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.image.copy()
                
                # 计算直方图
                hist = np.zeros(256)
                height, width = gray.shape
                for i in range(height):
                    for j in range(width):
                        hist[gray[i,j]] += 1
                
                # 归一化直方图
                total_pixels = height * width
                hist = hist / total_pixels
                
                # 计算最佳阈值
                max_variance = 0
                best_threshold = 0
                
                for t in range(256):
                    # 计算两个类的概率
                    w0 = np.sum(hist[:t])
                    w1 = np.sum(hist[t:])
                    
                    if w0 == 0 or w1 == 0:
                        continue
                    
                    # 计算两个类的平均值
                    mu0 = np.sum(np.arange(t) * hist[:t]) / w0
                    mu1 = np.sum(np.arange(t, 256) * hist[t:]) / w1
                    
                    # 计算类间方差
                    variance = w0 * w1 * (mu0 - mu1) ** 2
                    
                    # 更新最佳阈值
                    if variance > max_variance:
                        max_variance = variance
                        best_threshold = t
                
                # 使用最佳阈值进行分割
                return self.threshold_segmentation(best_threshold)
                
            except Exception as e:
                print(f"Otsu阈值分割失败: {str(e)}")
                return self.image
        return None

    def region_growing(self, seed_point, threshold):
        """区域生长法分割
        创建时间：2024-12-30
        作者：作者2
        Args:
            seed_point: 种子点坐标(x,y)
            threshold: 生长阈值
        Returns:
            分割后的二值图像数组
        """
        if self.image is not None:
            if len(self.image.shape) > 2:
                image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                image = self.image.copy()
                
            height, width = image.shape
            seed_x, seed_y = seed_point
            
            # 创建标记矩阵
            segmented = np.zeros((height, width), np.uint8)
            
            # 获取种子点的灰度值
            seed_value = float(image[seed_y, seed_x])
            
            # 定义4邻域
            dx = [-1, 1, 0, 0]
            dy = [0, 0, -1, 1]
            
            # 创建堆栈并添加种子点
            stack = [(seed_x, seed_y)]
            segmented[seed_y, seed_x] = 255
            
            while stack:
                x, y = stack.pop()
                for i in range(4):
                    new_x = x + dx[i]
                    new_y = y + dy[i]
                    
                    if (0 <= new_x < width and 0 <= new_y < height and 
                        segmented[new_y, new_x] == 0 and 
                        abs(float(image[new_y, new_x]) - seed_value) <= threshold):
                        segmented[new_y, new_x] = 255
                        stack.append((new_x, new_y))
                        
            return segmented
            
    def split_and_merge(self, min_size=8, threshold=30):
        """
        区域分裂与合并分割
        创建时间：2024-12-30
        作者：作者2
        功能：使用分裂合并算法对图像进行分割，输出黑白二值图像
        
        参数：
            min_size: 最小区大小，必须是2的幂次方
            threshold: 区域内像素差异阈值
            
        返回：
            ndarray: 分割后的二值图像
        """
        if self.image is not None:
            try:
                # 转换为灰度图像
                if len(self.image.shape) > 2:
                    gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.image.copy()
                
                # 确保图像尺寸是2的幂次方
                height, width = gray.shape
                size = max(width, height)
                size = 2 ** int(np.ceil(np.log2(size)))
                padded = np.zeros((size, size), dtype=np.uint8)
                padded[:height, :width] = gray
                
                # 创建标记数组
                labels = np.zeros((size, size), dtype=np.int32)
                next_label = 1
                
                def split(x, y, size):
                    """递归分裂区域"""
                    nonlocal next_label
                    
                    # 获取当前区域
                    region = padded[y:y+size, x:x+size]
                    
                    # 计算区域统计信息
                    mean_val = np.mean(region)
                    std_val = np.std(region)
                    
                    # 判断是否需要分裂
                    if size <= min_size or std_val <= threshold:
                        # 区域足够均匀，标记为同一区域
                        labels[y:y+size, x:x+size] = next_label
                        next_label += 1
                        return mean_val
                    
                    # 分裂为四个子区域
                    new_size = size // 2
                    m1 = split(x, y, new_size)  # 左
                    m2 = split(x + new_size, y, new_size)  # 右上
                    m3 = split(x, y + new_size, new_size)  # 左下
                    m4 = split(x + new_size, y + new_size, new_size)  # 右下
                    
                    return (m1 + m2 + m3 + m4) / 4
                
                def merge_regions():
                    """合并相似的相邻区域"""
                    # 获取所有区域标签
                    unique_labels = np.unique(labels)
                    if len(unique_labels) <= 1:
                        return
                        
                    # 计算每个区域的平均值
                    region_means = {}
                    for label in unique_labels:
                        if label == 0:
                            continue
                        mask = (labels == label)
                        region_means[label] = np.mean(padded[mask])
                    
                    # 合并相似区域
                    for y in range(1, size):
                        for x in range(1, size):
                            current = labels[y, x]
                            if current == 0:
                                continue
                                
                            # 检查上方和左方的区域
                            up = labels[y-1, x]
                            left = labels[y, x-1]
                            
                            if up != 0 and up != current:
                                if abs(region_means[current] - region_means[up]) <= threshold:
                                    labels[labels == up] = current
                                    region_means[current] = np.mean(padded[labels == current])
                                    
                            if left != 0 and left != current:
                                if abs(region_means[current] - region_means[left]) <= threshold:
                                    labels[labels == left] = current
                                    region_means[current] = np.mean(padded[labels == current])
                
                # 执行分裂
                print("开始分裂...")
                split(0, 0, size)
                
                # 执行合并
                print("开始合并...")
                merge_regions()
                print("合并完成")
                
                # 获取所有区域的平均灰度值
                region_means = {}
                for label in np.unique(labels):
                    if label == 0:
                        continue
                    mask = (labels == label)
                    region_means[label] = np.mean(padded[mask])
                
                # 创建二值图像
                binary = np.zeros((size, size), dtype=np.uint8)
                global_mean = np.mean(list(region_means.values()))
                
                # 根据区域平均值决定黑白
                for label in region_means:
                    binary[labels == label] = 255 if region_means[label] > global_mean else 0
                
                # 裁剪回原始尺寸
                binary = binary[:height, :width]
                
                # 转换为3通道图像
                result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                
                return result
                
            except Exception as e:
                print(f"区域分裂合并分割失败: {str(e)}")
                return self.image
        return None

    def load_cnn_model(self):
        """加载预训练的ResNet模型
        创建时间：2024-12-30
        作者：作者2
        Returns:
            bool: 模型加载是否成功
        """
        try:
            # 检查是否安装了torch
            try:
                import torch
                import torchvision.models as models
                import torchvision.transforms as transforms
                import os
            except ImportError:
                print("请先安装PyTorch和torchvision:")
                print("pip install torch torchvision")
                return False

            # 定义模型文件路径
            model_path = 'resnet50_model.pth'
            
            # 创建模型实例
            self.model = models.resnet50()
            
            # 检查是否存在本地模型文件
            if os.path.exists(model_path):
                print("正在加载本地模型...")
                # 加载本地模型文件
                self.model.load_state_dict(torch.load(model_path))
            else:
                print("正在下载预训练模型，这可能需要分钟...")
                # 下载预训练模型
                self.model = models.resnet50(pretrained=True)
                # 保存模型到本地
                torch.save(self.model.state_dict(), model_path)
                print("模型已保存到本地")
                
            self.model.eval()

            # 定义图像预处理
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # 类别映射和索引范围
            self.category_ranges = {
                'person': list(range(400, 406)),      # 人
                'dog': list(range(151, 269)),         # 狗的各种品种
                'cat': list(range(281, 294)),         # 猫的各种品种
                'car': list(range(817, 824)),         # 汽车、轿车等
                'phone': [487, 488, 489, 770],        # 手机、电话
                'laptop': [620, 621, 622],            # 笔记本电脑
                'chair': [423, 424, 425, 426],        # 椅子
                'bottle': [742, 743, 744, 745],       # 瓶子
                'food': list(range(924, 957)),        # 食物
                'bird': list(range(7, 24)),           # 鸟类
                'book': [497, 498, 499],              # 书本
                'flower': list(range(970, 980)),      # 花
                'tree': list(range(970, 980)),        # 树
                'building': list(range(660, 668))     # 建筑
            }
            
            # 我们关注的类别
            self.categories = list(self.category_ranges.keys())
                
            print("模型加载成功！")
            return True
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            return False
            
    def classify_image(self):
        """使用CNN对图像进行分类
        创建时间：2024-12-30
        作者：作者2
        """
        if self.image is None:
            print("请先加载图像")
            return None
            
        if not hasattr(self, 'model') or not hasattr(self, 'transform'):
            print("请先加载CNN模型")
            return None
            
        try:
            import torch
            from PIL import Image
            
            # 转换图像格式
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            
            try:
                # 预处理图像
                input_tensor = self.transform(image)
                input_batch = input_tensor.unsqueeze(0)
                
                # 进行预测
                with torch.no_grad():
                    output = self.model(input_batch)
                    
                # 获取预测结果
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                
                # 合并相关类别的概率
                category_probs = {cat: 0.0 for cat in self.categories}
                
                # 遍历所有类别，将相关类别的概率累加
                for idx, prob in enumerate(probabilities):
                    prob = float(prob)
                    # 检查每个类别的索引范围
                    for category, indices in self.category_ranges.items():
                        if idx in indices:
                            category_probs[category] += prob
                
                # 转换为结果列表
                results = [
                    {'category': cat, 'probability': prob}
                    for cat, prob in category_probs.items()
                    if prob > 0.01  # 只保留概率大于1%的结果
                ]
                
                # 按概率排序
                results.sort(key=lambda x: x['probability'], reverse=True)
                
                # 只返回前5个最可能的结果
                return results[:5]
            except Exception as e:
                print(f"图像处理或预测过程出错: {str(e)}")
                return None
                
        except Exception as e:
            print(f"分类失败: {str(e)}")
            return None

    def crop_image(self, x1, y1, x2, y2):
        """图像裁剪
        创建时间：2024-12-31
        作者：作者1
        Args:
            x1, y1: 裁剪区域左上角坐标
            x2, y2: 裁剪区域右下角坐标
        Returns:
            裁剪后的图像数组
        """
        if self.image is not None:
            try:
                height, width = self.image.shape[:2]
                
                # 确保坐标在有效范围内
                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))
                
                # 确保x1 < x2, y1 < y2
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # 裁剪图像
                return self.image[y1:y2, x1:x2].copy()
                
            except Exception as e:
                print(f"裁剪失败: {str(e)}")
                return self.image
        return None
