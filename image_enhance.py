import cv2
import numpy as np
from pathlib import Path
import os

# ===== 白平衡函数 =====
def simple_white_balance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    avg_a = np.average(lab[:, :, 1])
    avg_b = np.average(lab[:, :, 2])
    # 调整系数从 1.1 改为 0.5，并限制调整幅度
    lab[:, :, 1] -= ((avg_a - 128) * (lab[:, :, 0] / 255.0) * 0.5)  # 修改此处
    lab[:, :, 2] -= ((avg_b - 128) * (lab[:, :, 0] / 255.0) * 0.5)  # 修改此处
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ===== 调整图像大小 =====
def resize_image(image, width=None, height=None):
    h, w = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        ratio = height / h
        width = int(w * ratio)
    elif height is None:
        ratio = width / w
        height = int(h * ratio)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)

def wiener_deconvolution(image, psf, snr=300):
    result = np.zeros_like(image)
    for channel in range(3):  # BGR 分别处理
        blurred = image[:, :, channel].astype(np.float32)
        psf_fft = np.fft.fft2(psf, s=blurred.shape)
        blurred_fft = np.fft.fft2(blurred)
        psf_fft_conj = np.conj(psf_fft)
        wiener_filter = psf_fft_conj / (np.abs(psf_fft) ** 2 + 1 / snr)
        deconvolved_fft = wiener_filter * blurred_fft
        deconvolved = np.abs(np.fft.ifft2(deconvolved_fft))
        result[:, :, channel] = np.clip(deconvolved, 0, 255)
    return result.astype(np.uint8)

def make_circular_psf(size=15, radius=5):
    psf = np.zeros((size, size), dtype=np.float32)
    cv2.circle(psf, (size//2, size//2), radius, 1, -1)
    return psf / psf.sum()

def make_gaussian_psf(size=21, sigma=3):
    """生成一个中心对称的高斯模糊核，作为 PSF"""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

# ===== 去模糊（包括 unsharp） =====
def deblur_image(image, method="advanced", strength=30):
    original = image.copy()

    # if method == "advanced":
    #     denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    #     lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    #     l, a, b = cv2.split(lab)
    #     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    #     cl = clahe.apply(l)
    #     enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    #     sharpened = cv2.filter2D(enhanced, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
    #     gaussian = cv2.GaussianBlur(sharpened, (0, 0), 5)
    #     detail = cv2.addWeighted(sharpened, 1.5, gaussian, -0.5, 0)
    #     edge_preserve = cv2.edgePreservingFilter(detail, flags=1, sigma_s=60, sigma_r=0.4)
    #     return cv2.addWeighted(original, 1 - strength / 100, edge_preserve, strength / 100, 0)

    if method == "advanced":
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 11)  # 降低去噪强度
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 降低对比度增强强度
        cl = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        # 使用更温和的锐化核
        sharpened = cv2.filter2D(enhanced, -1, np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]]))
        # 移除高斯模糊和边缘保留滤波步骤
        return cv2.addWeighted(original, 0.8, sharpened, 0.2, 0)  # 降低混合强度
    
    elif method == "unsharp":
        blur = cv2.bilateralFilter(image, 9, 75, 75)
        detail = cv2.subtract(image, blur)
        enhanced = cv2.addWeighted(image, 1.0 + 2 * (strength / 100), detail, 2 * (strength / 100), 0)
        kernel = np.array([[-0.1, -0.1, -0.1], [-0.1, 2.0, -0.1], [-0.1, -0.1, -0.1]])
        return cv2.filter2D(enhanced, -1, kernel)
    
    elif method == "wiener":
        #psf = make_circular_psf(size=21, radius=6)
        psf = make_gaussian_psf(size=21, sigma=5)
        return wiener_deconvolution(image, psf, snr=300)

    return image

# ===== 新增的非AI增强流程 =====
def apply_enhancement_pipeline(image, apply_median=True, apply_clahe=True, apply_laplacian=True):
    """
    非AI图像增强流程：
    1. 中值滤波（去光晕伪影）
    2. CLAHE（提亮暗部）
    3. Laplacian（增强细节结构）
    """
    processed = image.copy()
    if apply_median:
        processed = cv2.medianBlur(processed, 5)
    if apply_clahe:
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        processed = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)
    if apply_laplacian:
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        laplacian_bgr = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
        processed = cv2.addWeighted(processed, 1.0, laplacian_bgr, 0.3, 0)
    return processed

# ===== 图像增强主流程 =====
def enhance_image(input_path, output_path, width=None, height=None, deblur=False, method="advanced", strength=50,
                  enhance_contrast=False, reduce_noise=False, median_filter=False, compare=True):
    image = cv2.imread(str(input_path))

    # Gamma 校正（添加到白平衡后）
    gamma = 0.9  # 小于1变亮，大于1变暗
    image = np.power(image / 255.0, gamma) * 255.0
    image = image.astype(np.uint8)

    if image is None:
        print(f"❌ 无法读取图像：{input_path}")
        return
    original = image.copy()

    if reduce_noise:
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    if median_filter:
        image = cv2.medianBlur(image, 5)

    if width or height:
        image = resize_image(image, width, height)
        original = resize_image(original, width, height)

    image = simple_white_balance(image)

    if deblur:
        image = deblur_image(image, method, strength)
    image = apply_enhancement_pipeline(
        image,
        apply_median=False,  # 如果已启用中值滤波（median_filter），此处关闭以避免重复
        apply_clahe=True,    # 替代原有的 enhance_contrast
        apply_laplacian=True
    )
    if enhance_contrast:
        # lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # l, a, b = cv2.split(lab)
        # cl = cv2.createCLAHE(1.5).apply(l)
        # image = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        pass
    # 保存图像
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    method_suffix = f"_{method}{strength}"
    base_name = output_path.stem
    ext = output_path.suffix

    compare_path = output_path.parent / f"{base_name}{method_suffix}_compare{ext}"
    enhanced_path = output_path.parent / f"{base_name}{method_suffix}_enhanced{ext}"

    if compare:
        if original.shape != image.shape:
            image = cv2.resize(image, (original.shape[1], original.shape[0]))
        comp = np.hstack((original, image))
        cv2.imwrite(str(compare_path), comp)
        print(f"🟩 对比图已保存: {compare_path}")

    cv2.imwrite(str(enhanced_path), image)
    print(f"✅ 增强图像已保存: {enhanced_path}")

# ===== 批处理入口 =====
def batch_process():
    input_dir = Path(args["batch"])
    output_dir = Path(args["output_dir"])
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpeg"))
    print(f"共检测到 {len(image_files)} 张图像")

    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] 处理 {img_path.name}...")
        out_path = output_dir / img_path.name
        enhance_image(
            input_path=img_path,
            output_path=out_path,
            width=args["width"],
            height=args["height"],
            deblur=args["deblur"],
            method=args["method"],
            strength=args["strength"],
            enhance_contrast=args["enhance_contrast"],
            reduce_noise=args["reduce_noise"],
            median_filter=args["median_filter"],
            compare=args["compare"]
        )

# ===== 参数配置 =====
args = {
    "batch": "./img",                     # 输入图片文件夹路径
    "output_dir": "output_images",        # 输出图片保存路径
    "width": None,                        # 目标宽度
    "height": None,                       # 目标高度
    "deblur": True,                       # 是否去模糊
    "method": "advanced",                  # 去模糊方法：advanced / unsharp
    "strength": 27,                       # 去模糊强度 1~100
    "enhance_contrast": False,             # 是否增强对比度
    "reduce_noise": True,                 # 是否去噪
    "median_filter": False,                # 是否进行中值滤波
    "compare": True                       # 是否输出原图对比
}

# ===== 主程序入口 =====
if __name__ == "__main__":
    batch_process()
