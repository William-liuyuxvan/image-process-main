import sys
import locale

# 设置默认编码
if sys.platform.startswith('win'):
    locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from image_processing import ImageProcessor
import matplotlib.pyplot as plt  # 用于显示直方图

class ImageProcessingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("图像处理系统")
        
        # 设置窗口关闭处理
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 创建图像处理器实例
        self.processor = ImageProcessor()
        
        # 设置最小窗口大小
        self.root.minsize(400, 300)
        
        # 创建界面组件
        self.create_widgets()
        
        # 保存原始图像
        self.original_image = None
        
        # 延迟绑定窗口大小改变事件，避免初始化时的触发
        self.root.after(100, self.bind_resize_event)
        
    def bind_resize_event(self):
        """延迟绑定窗口大小改变事件"""
        self.root.bind('<Configure>', self.on_window_resize)
        
    def create_widgets(self):
        # 创建主框架
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(expand=True, fill='both')
        
        # 创建菜单栏
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="打开", command=self.open_image)
        file_menu.add_command(label="保存", command=self.save_image)
        file_menu.add_command(label="还原", command=self.restore_image)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        
        # 几何变换菜单
        geometry_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="几何变换", menu=geometry_menu)
        geometry_menu.add_command(label="裁剪", command=self.crop_dialog)
        geometry_menu.add_command(label="缩放", command=self.scale_dialog)
        geometry_menu.add_command(label="平移", command=self.translate_dialog)
        geometry_menu.add_command(label="旋转", command=self.rotate_dialog)
        geometry_menu.add_command(label="翻转", command=self.flip_dialog)
        
        # 像素变换菜单
        pixel_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="像素变换", menu=pixel_menu)
        
        # 图像合成子菜单
        compose_menu = tk.Menu(pixel_menu, tearoff=0)
        pixel_menu.add_cascade(label="图像合成", menu=compose_menu)
        compose_menu.add_command(label="加法", command=self.addition_dialog)
        compose_menu.add_command(label="减法", command=self.subtraction_dialog)
        compose_menu.add_command(label="与运算", command=self.and_dialog)
        compose_menu.add_command(label="或运算", command=self.or_dialog)
        compose_menu.add_command(label="补运算", command=self.not_operation)
        compose_menu.add_command(label="异或", command=self.xor_dialog)
        
        # 灰度变换
        pixel_menu.add_command(label="转为灰度图", command=self.convert_to_gray)
        pixel_menu.add_command(label="亮度调整", command=self.brightness_dialog)
        
        # 直方图操作
        pixel_menu.add_command(label="显示直方图", command=self.show_histogram)
        pixel_menu.add_command(label="直方图均衡化", command=self.equalize_hist)
        
        # 图像去噪菜单
        denoise_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="图像去噪", menu=denoise_menu)
        denoise_menu.add_command(label="中值滤波", command=self.median_filter_dialog)
        denoise_menu.add_command(label="均值滤波", command=self.mean_filter_dialog)
        denoise_menu.add_command(label="高斯滤波", command=self.gaussian_filter_dialog)
        denoise_menu.add_command(label="频域滤波", command=self.frequency_filter_dialog)
        
        # 图像锐化菜单
        sharpen_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="图像锐化", menu=sharpen_menu)
        sharpen_menu.add_command(label="拉普拉斯锐化", command=self.laplacian_sharpen)
        sharpen_menu.add_command(label="USM锐化", command=self.unsharp_masking_dialog)
        
        # 边缘检测菜单
        edge_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="边缘检测", menu=edge_menu)
        edge_menu.add_command(label="Sobel算子", command=self.sobel_edge)
        edge_menu.add_command(label="Canny算子", command=self.canny_edge_dialog)
        edge_menu.add_command(label="Prewitt算子", command=self.prewitt_edge)
        
        # 图像分割菜单
        segment_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="图像分割", menu=segment_menu)
        segment_menu.add_command(label="阈值分割", command=self.threshold_dialog)
        segment_menu.add_command(label="Otsu阈值分割", command=self.otsu_threshold)
        segment_menu.add_command(label="区域生长", command=self.region_growing_dialog)
        segment_menu.add_command(label="区域分裂与合并", command=self.split_merge_dialog)
        
        # 在现有菜单后加CNN分类菜单
        cnn_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="CNN分类", menu=cnn_menu)
        cnn_menu.add_command(label="加载模型", command=self.load_model)
        cnn_menu.add_command(label="图像分类", command=self.classify_dialog)
        
        # 创建图像显示区域
        self.canvas = tk.Canvas(self.main_frame)
        self.canvas.pack(expand=True, fill='both', padx=5, pady=5)
        
    def on_window_resize(self, event):
        """窗口大小改变时的回调函数"""
        # 只处理主窗口的大小改变事件
        if event.widget == self.root:
            # 添加小延迟，等待窗口大小稳定
            self.root.after(100, self.delayed_resize)
            
    def delayed_resize(self):
        """延迟处理窗口大小改变"""
        if self.processor.image is not None:
            self.display_image()
            
    def display_image(self):
        if self.processor.image is not None:
            try:
                # 保存原始图像
                if self.original_image is None:
                    self.original_image = self.processor.image.copy()
                
                # 获取画布大小
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                # 确保有最小显示区域
                if canvas_width < 10 or canvas_height < 10:
                    return
                    
                # 检查图像是否有效
                if self.processor.image.size == 0:
                    messagebox.showerror("错误", "图像数据无效")
                    return
                    
                # 转换图像格式
                image = cv2.cvtColor(self.processor.image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                
                # 计算缩放比例，保持图像原始大小
                image_width, image_height = image.size
                if image_width <= canvas_width and image_height <= canvas_height:
                    # 如果图像小于画布，直接显示原始大小
                    display_width = image_width
                    display_height = image_height
                else:
                    # 如果图像大于画布，按比例缩小
                    scale = min((canvas_width - 20) / image_width, 
                              (canvas_height - 20) / image_height)
                    display_width = int(image_width * scale)
                    display_height = int(image_height * scale)
                
                # 调整图像大小
                if display_width > 0 and display_height > 0:
                    image = image.resize((display_width, display_height), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(image=image)
                    
                    # 计算居中位置
                    x = (canvas_width - display_width) // 2
                    y = (canvas_height - display_height) // 2
                    
                    # 更新画布
                    self.canvas.delete("all")
                    self.canvas.create_image(x, y, anchor='nw', image=photo)
                    self.canvas.image = photo  # 保持引用
                    
            except Exception as e:
                messagebox.showerror("错误", f"显示图像时出错: {str(e)}")
    
    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="打开图像",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            try:
                self.processor.load_image(file_path)
                if self.processor.image is not None:
                    # 保存原始图像
                    self.original_image = self.processor.image.copy()
                    self.display_image()
                else:
                    messagebox.showerror("错误", "无法加载图像")
            except Exception as e:
                messagebox.showerror("错误", f"打开图像时出错: {str(e)}")
            
    def save_image(self):
        if self.processor.image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png")
            if file_path:
                self.processor.save_image(file_path)
                
    # 几何变换对话框
    def scale_dialog(self):
        if self.processor.image is None:
            messagebox.showerror("错误", "请先打开图片")
            return
            
        dialog = tk.Toplevel(self.root)
        dialog.title("缩放")
        
        tk.Label(dialog, text="X缩放比例:").grid(row=0, column=0)
        scale_x = tk.Entry(dialog)
        scale_x.insert(0, "2.0")  # 默认放大2倍
        scale_x.grid(row=0, column=1)
        
        tk.Label(dialog, text="Y缩放比例:").grid(row=1, column=0)
        scale_y = tk.Entry(dialog)
        scale_y.insert(0, "2.0")  # 默认放大2倍
        scale_y.grid(row=1, column=1)
        
        def apply():
            try:
                sx = float(scale_x.get())
                sy = float(scale_y.get())
                
                if sx <= 0 or sy <= 0:
                    messagebox.showerror("错误", "缩放比例必须大于0")
                    return
                
                # 执行缩放
                self.processor.scale_image(sx, sy)
                # 更新显示
                self.display_image()
                dialog.destroy()
                
            except ValueError:
                messagebox.showerror("错误", "请输入有效的数值")
                
        tk.Button(dialog, text="确定", command=apply).grid(row=2, column=0, columnspan=2)

    def addition_dialog(self):
        if self.processor.image is None:
            messagebox.showerror("错误", "请先打开一张图片")
            return
            
        file_path = filedialog.askopenfilename(
            title="选择要叠加的图片",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            image2 = cv2.imread(file_path)
            if image2 is not None:
                self.processor.image = self.processor.image_addition(image2)
                self.display_image()
            else:
                messagebox.showerror("错误", "无法加载选择的图片")

    def brightness_dialog(self):
        """亮度和对比度调整对话框"""
        if self.processor.image is None:
            messagebox.showerror("错误", "请先打开图片")
            return
            
        dialog = tk.Toplevel(self.root)
        dialog.title("亮度和对比度调整")
        
        # 保存原始图像
        original_image = self.processor.image.copy()
        
        # 亮度输入框 (-255 到 255)
        tk.Label(dialog, text="亮度调整(-255到255):").grid(row=0, column=0)
        brightness = tk.Entry(dialog, width=10)
        brightness.insert(0, "0")
        brightness.grid(row=0, column=1)
        
        # 对比度输入框 (0 到 3)
        tk.Label(dialog, text="对比度调整(0到3):").grid(row=1, column=0)
        contrast = tk.Entry(dialog, width=10)
        contrast.insert(0, "1.0")
        contrast.grid(row=1, column=1)
        
        def preview():
            try:
                b = float(brightness.get())
                c = float(contrast.get())
                
                # 检查输入范围
                if not (-255 <= b <= 255):
                    messagebox.showerror("错误", "亮度值必须在-255到255之间")
                    return
                if not (0 <= c <= 3):
                    messagebox.showerror("错误", "对比度值必须在0到3之间")
                    return
                
                # 应用调整
                result = self.processor.adjust_brightness_contrast(b, c)
                if result is not None:
                    self.processor.image = result
                    self.display_image()
            except ValueError:
                messagebox.showerror("错误", "请输入有效的数值")
        
        def apply():
            preview()
            dialog.destroy()
        
        def cancel():
            self.processor.image = original_image
            self.display_image()
            dialog.destroy()
        
        # 按钮框架
        button_frame = tk.Frame(dialog)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        tk.Button(button_frame, text="预览", command=preview).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="确定", command=apply).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="取消", command=cancel).pack(side=tk.LEFT, padx=5)

    def median_filter_dialog(self):
        """中值滤波"""
        if self.processor.image is None:
            messagebox.showerror("错误", "请先打开图片")
            return
            
        # 直接应用中值滤波，使用默认参数
        result = self.processor.median_filter()
        if result is not None:
            self.processor.image = result
            self.display_image()

    def frequency_filter_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("频域滤波")
        
        tk.Label(dialog, text="截止频率:").grid(row=0, column=0)
        d0 = tk.Entry(dialog)
        d0.insert(0, "30")
        d0.grid(row=0, column=1)
        
        filter_type = tk.StringVar(value="lowpass")
        tk.Radiobutton(dialog, text="低通滤波", variable=filter_type, 
                      value="lowpass").grid(row=1, column=0)
        tk.Radiobutton(dialog, text="高通滤波", variable=filter_type, 
                      value="highpass").grid(row=1, column=1)
        
        def apply():
            try:
                d0_value = int(d0.get())
                self.processor.image = self.processor.frequency_filter(d0_value, filter_type.get())
                self.display_image()
                dialog.destroy()
            except ValueError:
                messagebox.showerror("错误", "请输入有效的整数")
                
        tk.Button(dialog, text="确定", command=apply).grid(row=2, column=0, columnspan=2)

    def laplacian_sharpen(self):
        if self.processor.image is not None:
            self.processor.image = self.processor.laplacian_sharpen()
            self.display_image()
            
    def unsharp_masking_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("USM锐化")
        
        tk.Label(dialog, text="数量(amount):").grid(row=0, column=0)
        amount = tk.Entry(dialog)
        amount.insert(0, "1.0")
        amount.grid(row=0, column=1)
        
        tk.Label(dialog, text="半径(radius):").grid(row=1, column=0)
        radius = tk.Entry(dialog)
        radius.insert(0, "5")
        radius.grid(row=1, column=1)
        
        tk.Label(dialog, text="阈值(threshold):").grid(row=2, column=0)
        threshold = tk.Entry(dialog)
        threshold.insert(0, "0")
        threshold.grid(row=2, column=1)
        
        def apply():
            try:
                a = float(amount.get())
                r = int(radius.get())
                t = int(threshold.get())
                self.processor.image = self.processor.unsharp_masking(a, r, t)
                self.display_image()
                dialog.destroy()
            except ValueError:
                messagebox.showerror("错误", "请输入有效的数值")
                
        tk.Button(dialog, text="确定", command=apply).grid(row=3, column=0, columnspan=2)
        
    def sobel_edge(self):
        if self.processor.image is not None:
            self.processor.image = self.processor.sobel_edge()
            self.display_image()
            
    def canny_edge_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Canny边缘检测")
        
        tk.Label(dialog, text="阈值1:").grid(row=0, column=0)
        threshold1 = tk.Entry(dialog)
        threshold1.insert(0, "100")
        threshold1.grid(row=0, column=1)
        
        tk.Label(dialog, text="阈值2:").grid(row=1, column=0)
        threshold2 = tk.Entry(dialog)
        threshold2.insert(0, "200")
        threshold2.grid(row=1, column=1)
        
        def apply():
            try:
                t1 = int(threshold1.get())
                t2 = int(threshold2.get())
                self.processor.image = self.processor.canny_edge(t1, t2)
                self.display_image()
                dialog.destroy()
            except ValueError:
                messagebox.showerror("错误", "请输入有效的整数")
                
        tk.Button(dialog, text="确定", command=apply).grid(row=2, column=0, columnspan=2)
        
    def prewitt_edge(self):
        if self.processor.image is not None:
            self.processor.image = self.processor.prewitt_edge()
            self.display_image()

    def threshold_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("阈值分割")
        
        tk.Label(dialog, text="阈值:").grid(row=0, column=0)
        threshold = tk.Entry(dialog)
        threshold.insert(0, "128")
        threshold.grid(row=0, column=1)
        
        tk.Label(dialog, text="最大值:").grid(row=1, column=0)
        max_val = tk.Entry(dialog)
        max_val.insert(0, "255")
        max_val.grid(row=1, column=1)
        
        def apply():
            try:
                t = int(threshold.get())
                m = int(max_val.get())
                self.processor.image = self.processor.threshold_segmentation(t, m)
                self.display_image()
                dialog.destroy()
            except ValueError:
                messagebox.showerror("错误", "请输入有效的整数")
                
        tk.Button(dialog, text="确定", command=apply).grid(row=2, column=0, columnspan=2)
        
    def otsu_threshold(self):
        if self.processor.image is not None:
            self.processor.image = self.processor.otsu_threshold()
            self.display_image()
            
    def region_growing_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("区域生长")
        
        tk.Label(dialog, text="阈值:").grid(row=0, column=0)
        threshold = tk.Entry(dialog)
        threshold.insert(0, "10")
        threshold.grid(row=0, column=1)
        
        tk.Label(dialog, text="请在图像上点击选择种子点").grid(row=1, column=0, columnspan=2)
        
        def on_click(event):
            try:
                t = float(threshold.get())
                # 修正坐标转换
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                image_height, image_width = self.processor.image.shape[:2]
                
                # 计算图像在画布上的实际位置和大小
                scale = min((canvas_width-10)/image_width, (canvas_height-10)/image_height)
                new_width = int(image_width * scale)
                new_height = int(image_height * scale)
                x_offset = (canvas_width - new_width) // 2
                y_offset = (canvas_height - new_height) // 2
                
                # 转换画布坐标到图像坐标
                image_x = int((event.x - x_offset) / scale)
                image_y = int((event.y - y_offset) / scale)
                
                # 检查坐标是否在图像范围内
                if 0 <= image_x < image_width and 0 <= image_y < image_height:
                    seed_point = (image_x, image_y)
                    self.processor.image = self.processor.region_growing(seed_point, t)
                    self.display_image()
                    dialog.destroy()
                else:
                    messagebox.showerror("错误", "请在图像区域内点击")
            except ValueError:
                messagebox.showerror("错误", "请输入有效的数值")
                
        self.canvas.bind('<Button-1>', on_click)
        dialog.protocol("WM_DELETE_WINDOW", lambda: self.canvas.unbind('<Button-1>'))
        
    def split_merge_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("区域分裂与合并")
        
        tk.Label(dialog, text="小区域大小:").grid(row=0, column=0)
        min_size = tk.Entry(dialog)
        min_size.insert(0, "8")
        min_size.grid(row=0, column=1)
        
        tk.Label(dialog, text="阈值:").grid(row=1, column=0)
        threshold = tk.Entry(dialog)
        threshold.insert(0, "30")
        threshold.grid(row=1, column=1)
        
        def apply():
            try:
                ms = int(min_size.get())
                t = float(threshold.get())
                self.processor.image = self.processor.split_and_merge(ms, t)
                self.display_image()
                dialog.destroy()
            except ValueError:
                messagebox.showerror("错误", "请输入有效的数值")
                
        tk.Button(dialog, text="确定", command=apply).grid(row=2, column=0, columnspan=2)

    def load_model(self):
        """加载CNN模型"""
        try:
            import torch
        except ImportError:
            messagebox.showerror("错误", "请先安装PyTorch:\npip install torch torchvision")
            return
            
        try:
            result = self.processor.load_cnn_model()
            if result:
                messagebox.showinfo("成功", "模型加载成功！")
            else:
                messagebox.showerror("错误", "模型加载失败！请检查控制台输出了解详细信息。")
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败: {str(e)}")
            
    def classify_dialog(self):
        """显示分类结果对话框"""
        if self.processor.image is None:
            messagebox.showerror("错误", "请先打开一张图片！")
            return
            
        if not hasattr(self.processor, 'model'):
            messagebox.showerror("错误", "请先加载模型！")
            return
            
        results = self.processor.classify_image()
        if results:
            dialog = tk.Toplevel(self.root)
            dialog.title("分类结果")
            
            # 显示预测结果
            for i, result in enumerate(results):
                category = result['category']
                prob = result['probability']
                tk.Label(dialog, 
                        text=f"{i+1}. {category}: {prob:.2%}").grid(
                        row=i, column=0, sticky='w', padx=5, pady=2)
        else:
            messagebox.showerror("错误", "分类失败！请检查控制台输出了解详细信息。")

    def translate_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("平移")
        
        tk.Label(dialog, text="X方向平移:").grid(row=0, column=0)
        dx = tk.Entry(dialog)
        dx.insert(0, "0")
        dx.grid(row=0, column=1)
        
        tk.Label(dialog, text="Y方向平移:").grid(row=1, column=0)
        dy = tk.Entry(dialog)
        dy.insert(0, "0")
        dy.grid(row=1, column=1)
        
        def apply():
            try:
                x = int(dx.get())
                y = int(dy.get())
                self.processor.image = self.processor.translate_image(x, y)
                self.display_image()
                dialog.destroy()
            except ValueError:
                messagebox.showerror("错误", "请输入有效的整数")
                
        tk.Button(dialog, text="确定", command=apply).grid(row=2, column=0, columnspan=2)

    def rotate_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("旋转")
        
        tk.Label(dialog, text="旋转角度:").grid(row=0, column=0)
        angle = tk.Entry(dialog)
        angle.insert(0, "0")
        angle.grid(row=0, column=1)
        
        def apply():
            try:
                a = float(angle.get())
                self.processor.image = self.processor.rotate_image(a)
                self.display_image()
                dialog.destroy()
            except ValueError:
                messagebox.showerror("错误", "请输入有效的数值")
                
        tk.Button(dialog, text="确定", command=apply).grid(row=1, column=0, columnspan=2)

    def flip_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("翻转")
        
        flip_type = tk.IntVar(value=0)
        tk.Radiobutton(dialog, text="上下翻转", variable=flip_type, 
                      value=0).grid(row=0, column=0)
        tk.Radiobutton(dialog, text="左右翻转", variable=flip_type, 
                      value=1).grid(row=0, column=1)
        tk.Radiobutton(dialog, text="上下左右翻转", variable=flip_type, 
                      value=-1).grid(row=0, column=2)
        
        def apply():
            self.processor.image = self.processor.flip_image(flip_type.get())
            self.display_image()
            dialog.destroy()
                
        tk.Button(dialog, text="确定", command=apply).grid(row=1, column=0, columnspan=3)

    def subtraction_dialog(self):
        if self.processor.image is None:
            messagebox.showerror("错误", "请先打开一张图片")
            return
            
        file_path = filedialog.askopenfilename(
            title="选择要减去的图片",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            image2 = cv2.imread(file_path)
            if image2 is not None:
                self.processor.image = self.processor.image_subtraction(image2)
                self.display_image()
            else:
                messagebox.showerror("错误", "无法加载选择的图片")

    def and_dialog(self):
        if self.processor.image is None:
            messagebox.showerror("错误", "请先打开一张图片")
            return
            
        file_path = filedialog.askopenfilename(
            title="选择进行与运算的图片",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            image2 = cv2.imread(file_path)
            if image2 is not None:
                # 调整第二张图片的大小以匹配第一张图片
                if self.processor.image.shape != image2.shape:
                    image2 = cv2.resize(image2, (self.processor.image.shape[1], 
                                               self.processor.image.shape[0]))
                self.processor.image = self.processor.image_and(image2)
                self.display_image()
            else:
                messagebox.showerror("错误", "无法加载选择的图片")

    def or_dialog(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image2 = cv2.imread(file_path)
            if image2 is not None:
                self.processor.image = self.processor.image_or(image2)
                self.display_image()

    def not_operation(self):
        if self.processor.image is not None:
            self.processor.image = self.processor.image_not()
            self.display_image()

    def xor_dialog(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image2 = cv2.imread(file_path)
            if image2 is not None:
                self.processor.image = self.processor.image_xor(image2)
                self.display_image()

    def convert_to_gray(self):
        if self.processor.image is not None:
            self.processor.image = self.processor.to_grayscale()
            self.display_image()

    def show_histogram(self):
        if self.processor.image is not None:
            import matplotlib.pyplot as plt
            hist = self.processor.calculate_histogram()
            colors = ('b', 'g', 'r')
            plt.figure()
            for i, color in enumerate(colors):
                plt.plot(hist[i], color=color)
            plt.title('图像直方图')
            plt.show()

    def equalize_hist(self):
        if self.processor.image is not None:
            self.processor.image = self.processor.equalize_histogram()
            self.display_image()

    def mean_filter_dialog(self):
        """均值滤波"""
        if self.processor.image is None:
            messagebox.showerror("错误", "请先打开图片")
            return
            
        # 直接应用均值滤波，使用默认参数
        result = self.processor.mean_filter(3)  # 使用3x3的核
        if result is not None:
            self.processor.image = result
            self.display_image()

    def gaussian_filter_dialog(self):
        """高斯滤波"""
        if self.processor.image is None:
            messagebox.showerror("错误", "请先打开图片")
            return
            
        # 直接应用高斯滤波，使用默认参数
        result = self.processor.gaussian_filter(3)  # 使用3x3的核
        if result is not None:
            self.processor.image = result
            self.display_image()

    def on_closing(self):
        """窗口关闭时的处理"""
        try:   
            # 清理资源
            if hasattr(self.processor, 'model'):
                del self.processor.model
            if hasattr(self, 'original_image'):
                del self.original_image
            plt.close('all')  # 关闭所有matplotlib窗口
        finally:
            self.root.destroy()

    def restore_image(self):
        """还原到原始图像"""
        if self.original_image is not None:
            try:
                # 恢复原始图像
                self.processor.image = self.original_image.copy()
                self.display_image()
            except Exception as e:
                messagebox.showerror("错误", f"还原图像时出错: {str(e)}")
        else:
            messagebox.showwarning("警告", "没有可还原的图像")

    def crop_dialog(self):
        """图像裁剪对话框"""
        if self.processor.image is None:
            messagebox.showerror("错误", "请先打开图片")
            return
            
        dialog = tk.Toplevel(self.root)
        dialog.title("图像裁剪")
        
        # 创建画布用于显示裁剪区域
        canvas = tk.Canvas(dialog)
        canvas.pack(expand=True, fill='both', padx=5, pady=5)
        
        # 显示图像
        image = cv2.cvtColor(self.processor.image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        # 调整图像大小以适应画布
        canvas_width = 400
        canvas_height = 300
        scale = min(canvas_width/image.width, canvas_height/image.height)
        display_width = int(image.width * scale)
        display_height = int(image.height * scale)
        
        image = image.resize((display_width, display_height), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image=image)
        canvas.create_image(0, 0, anchor='nw', image=photo)
        canvas.image = photo
        
        # 用于存储裁剪区域坐标
        coords = {'x1': None, 'y1': None, 'x2': None, 'y2': None}
        rect_id = None
        
        def start_crop(event):
            coords['x1'] = event.x
            coords['y1'] = event.y
            
        def draw_rect(event):
            nonlocal rect_id
            coords['x2'] = event.x
            coords['y2'] = event.y
            
            # 删除旧的矩形
            if rect_id:
                canvas.delete(rect_id)
            
            # 绘制新的矩形
            rect_id = canvas.create_rectangle(
                coords['x1'], coords['y1'], 
                coords['x2'], coords['y2'],
                outline='red'
            )
            
        def apply_crop():
            if all(coords.values()):
                # 转换回原始图像坐标
                x1 = int(coords['x1'] / scale)
                y1 = int(coords['y1'] / scale)
                x2 = int(coords['x2'] / scale)
                y2 = int(coords['y2'] / scale)
                
                # 应用裁剪
                result = self.processor.crop_image(x1, y1, x2, y2)
                if result is not None:
                    self.processor.image = result
                    self.display_image()
                dialog.destroy()
        
        # 绑定鼠标事件
        canvas.bind('<Button-1>', start_crop)
        canvas.bind('<B1-Motion>', draw_rect)
        
        # 按钮
        tk.Button(dialog, text="确定", command=apply_crop).pack(pady=5)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingGUI(root)
    root.mainloop()









