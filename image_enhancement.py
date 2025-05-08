import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import signal
from numpy.fft import fft2, ifft2, fftshift, ifftshift

# 用于显示图像的辅助函数
def show_images(images, titles, rows=1, cols=None):
    """
    显示多张图片并排比较
    """
    if cols is None:
        cols = len(images) // rows + (1 if len(images) % rows else 0)
    
    plt.figure(figsize=(4*cols, 4*rows))
    
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i+1)
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 1. 拉普拉斯锐化滤波器
def laplacian_sharpening(image):
    """
    使用拉普拉斯算子进行图像锐化
    """
    # 转换为灰度图像如果是彩色图像
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 构建拉普拉斯算子
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # 归一化拉普拉斯结果，使其值在合理范围内
    laplacian = cv2.normalize(laplacian, None, 0, 1, cv2.NORM_MINMAX)
    
    # 如果是彩色图像，对每个通道应用锐化
    if len(image.shape) == 3:
        sharpened = np.zeros_like(image, dtype=float)
        for i in range(3):
            sharpened[:,:,i] = image[:,:,i] - 0.5 * laplacian  # 减去拉普拉斯，这实际上是锐化操作
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    else:
        sharpened = gray - 0.5 * laplacian
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened

# 2. 高通滤波器
def high_pass_filter(image, cutoff_freq=30):
    """
    使用高通滤波器增强图像细节
    """
    # 转换为灰度图像如果是彩色图像
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 傅里叶变换
    f = fft2(gray)
    fshift = fftshift(f)
    
    # 创建高通滤波器
    rows, cols = gray.shape
    crow, ccol = rows//2, cols//2
    
    # 创建掩码，中心区域为0，周围为1
    mask = np.ones((rows, cols), np.uint8)
    mask[crow-cutoff_freq:crow+cutoff_freq, ccol-cutoff_freq:ccol+cutoff_freq] = 0
    
    # 应用掩码并进行逆变换
    fshift = fshift * mask
    f_ishift = ifftshift(fshift)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # 归一化以便显示
    img_high = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    
    # 如果是彩色图像，合并高通滤波结果
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for i in range(3):
            result[:,:,i] = image[:,:,i] * 0.7 + img_high * 0.3  # 混合原图和高通滤波结果
        result = np.clip(result, 0, 255).astype(np.uint8)
    else:
        result = img_high.astype(np.uint8)
    
    return result

# 3. 非锐化掩蔽(Unsharp Masking)
def unsharp_masking(image, sigma=1.0, strength=1.5):
    """
    使用非锐化掩蔽技术增强图像锐度
    """
    # 如果是彩色图像，分别处理每个通道
    if len(image.shape) == 3:
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    else:
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    
    return sharpened

# 4. 傅里叶变换去模糊
def fourier_deblur(image, kernel_size=15, strength=0.1):
    """
    使用傅里叶变换进行简单的去模糊处理
    """
    # 转换为灰度图像如果是彩色图像
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 傅里叶变换
    f = fft2(gray)
    fshift = fftshift(f)
    
    # 创建一个增强高频的滤波器
    rows, cols = gray.shape
    crow, ccol = rows//2, cols//2
    
    # 创建高频增强滤波器
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    distance = np.sqrt(x*x + y*y)
    high_pass = 1 + strength * (distance / (rows/2))
    high_pass = np.minimum(high_pass, 2.0)  # 限制增强幅度
    
    # 应用滤波器
    fshift = fshift * high_pass
    
    # 逆变换
    f_ishift = ifftshift(fshift)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # 归一化
    img_sharp = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    
    # 如果是彩色图像，将处理后的灰度图与原始彩色图混合
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:,:,2] = img_sharp  # 将亮度通道替换为锐化后的图像
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        result = img_sharp.astype(np.uint8)
    
    return result

# 5. 维纳滤波(Wiener Filter)
def wiener_filter(image, noise_power=0.01, kernel_size=5):
    """
    实现维纳滤波去模糊
    """
    # 模拟一个模糊核
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size*kernel_size)
    
    # 分通道处理图像
    if len(image.shape) == 3:
        restored = np.zeros_like(image, dtype=np.float32)
        for i in range(3):
            channel = image[:,:,i].astype(np.float32)
            # FFT变换图像和点扩散函数(PSF)
            img_fft = fft2(channel)
            psf_fft = fft2(kernel, s=channel.shape)
            
            # 共轭复数
            psf_fft_conj = np.conj(psf_fft)
            
            # 维纳滤波公式
            result = np.real(ifft2(img_fft * psf_fft_conj / (np.abs(psf_fft)**2 + noise_power)))
            
            # 归一化结果
            result = np.clip(result, 0, 255)
            restored[:,:,i] = result
    else:
        img_float = image.astype(np.float32)
        img_fft = fft2(img_float)
        psf_fft = fft2(kernel, s=image.shape)
        psf_fft_conj = np.conj(psf_fft)
        result = np.real(ifft2(img_fft * psf_fft_conj / (np.abs(psf_fft)**2 + noise_power)))
        restored = np.clip(result, 0, 255)
    
    return restored.astype(np.uint8)

# 6. 双三次插值调整大小
def bicubic_resize(image, new_size):
    """
    使用双三次插值调整图像大小
    """
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    return resized

# 7. Lanczos重采样
def lanczos_resize(image, new_size):
    """
    使用Lanczos重采样调整图像大小
    """
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
    return resized

# 主函数，展示所有方法的效果
def enhance_image(input_path, output_path=None):
    """
    读取图像并应用所有增强方法
    """
    # 读取图像
    original = cv2.imread(input_path)
    if original is None:
        print(f"无法读取图像：{input_path}")
        return
    
    # 应用各种增强方法
    laplacian_sharp = laplacian_sharpening(original)
    high_pass = high_pass_filter(original)
    unsharp = unsharp_masking(original)
    fourier = fourier_deblur(original)
    wiener = wiener_filter(original)
    
    # 调整图像大小示例
    new_size = (original.shape[1]*2, original.shape[0]*2)  # 放大2倍
    bicubic = bicubic_resize(original, new_size)
    lanczos = lanczos_resize(original, new_size)
    
    # 显示原始增强效果
    show_images(
        [original, laplacian_sharp, high_pass, unsharp, fourier, wiener],
        ["original", "laplacian_sharp", "high_pass", "unsharp", "fourier", "wiener"], 
        rows=2
    )
    
    # 显示调整大小的结果(这里显示的是调整后图像的一部分，因为整个图像太大)
    crop_size = (min(500, original.shape[1]), min(500, original.shape[0]))
    crop_original = original[:crop_size[1], :crop_size[0]]
    crop_bicubic = bicubic[:crop_size[1]*2, :crop_size[0]*2]
    crop_lanczos = lanczos[:crop_size[1]*2, :crop_size[0]*2]
    
    show_images(
        [crop_original, crop_bicubic, crop_lanczos],
        ["原始图像(裁剪)", "双三次插值(裁剪)", "Lanczos重采样(裁剪)"],
        rows=1
    )
    
    # 保存结果
    if output_path:
        cv2.imwrite(output_path + "_laplacian.jpg", laplacian_sharp)
        cv2.imwrite(output_path + "_highpass.jpg", high_pass)
        cv2.imwrite(output_path + "_unsharp.jpg", unsharp)
        cv2.imwrite(output_path + "_fourier.jpg", fourier)
        cv2.imwrite(output_path + "_wiener.jpg", wiener)
        cv2.imwrite(output_path + "_bicubic.jpg", bicubic)
        cv2.imwrite(output_path + "_lanczos.jpg", lanczos)
        print(f"所有增强效果已保存到 {output_path}_*.jpg")

# 使用示例
if __name__ == "__main__":
    # 替换为您的图像路径
    input_image = "./img/deblur-4.jpg"
    output_prefix = "enhanced"
    enhance_image(input_image, output_prefix)