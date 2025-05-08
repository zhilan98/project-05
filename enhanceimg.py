import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift
import os
from skimage.restoration import richardson_lucy

def resize_image(image, width=None, height=None):
    if width is None and height is None:
        return image
    h, w = image.shape[:2]
    if width is not None and height is not None:
        return cv2.resize(image, (width, height))
    elif width is not None:
        ratio = width / w
        return cv2.resize(image, (width, int(h * ratio)))
    else:
        ratio = height / h
        return cv2.resize(image, (int(w * ratio), height))

def show_comparison(original, enhanced, title1="original_image", title2="after_enhanced"):
    """Show comparison chart"""
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title(title1)
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    plt.title(title2)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def evaluate_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    score = laplacian.var()  # The larger the variance, the clearer it is.
    return score

def laplacian_enhance(image):
    lap = cv2.Laplacian(image, cv2.CV_64F)
    lap = np.clip(lap, -255, 255).astype(np.int16)  
    image_int = image.astype(np.int16)
    enhanced = image_int + lap
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return enhanced

def light_sharpen(image):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(image, 1.3, blurred, -0.5, 0)
    return sharpened

def richardson_deblur(image, psf, iterations=10):
    result = np.zeros_like(image)
    for i in range(3):
        result[:, :, i] = richardson_lucy(image[:, :, i], psf, iterations=iterations)
    return np.clip(result * 255, 0, 255).astype(np.uint8)

def clahe_then_sharpen(image, strength=1.8, sigma=1.0):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    image_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    
    blurred = cv2.GaussianBlur(image_eq, (0, 0), sigma)
    sharpened = cv2.addWeighted(image_eq, 1 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def edge_weighted_unsharp(image, strength=1.5, sigma=1.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    edge_mask = np.abs(edges) / 255.0
    edge_mask = np.clip(edge_mask, 0.3, 1.0) 
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    detail = image.astype(np.float32) - blurred.astype(np.float32)
    enhanced = image.astype(np.float32) + detail * strength * edge_mask[..., np.newaxis]
    return np.clip(enhanced, 0, 255).astype(np.uint8)

def strong_unsharp_masking(image, sigma, strength, iterations, edge_boost, edge_weight):
    """Enhanced unsharp masking, using multiple iterations to enhance the effect"""

    result = image.copy()
    for _ in range(iterations):
        blurred = cv2.GaussianBlur(result, (0, 0), sigma)
        result = cv2.addWeighted(result, 1.0 + strength, blurred, -strength, 0)
        result = np.clip(result, 0, 255).astype(np.uint8)
    if edge_boost:
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY if result.shape[2] == 3 else cv2.COLOR_BGR2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB if result.shape[2] == 3 else cv2.COLOR_GRAY2BGR)
        result = cv2.addWeighted(result, 1.0, edges_colored, edge_weight, 0)
        result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def unsharp_in_y_channel(image, sigma=1.0, strength=2.5):
    """Sharpen only the brightness channel to prevent color distortion"""

    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(yuv)
    blurred_y = cv2.GaussianBlur(y, (0, 0), sigma)
    sharpened_y = cv2.addWeighted(y, 1.0 + strength, blurred_y, -strength, 0)
    sharpened_y = np.clip(sharpened_y, 0, 255).astype(np.uint8)
    merged = cv2.merge([sharpened_y, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

def multiscale_unsharp(image, sigmas=[1.0, 2.0, 3.0], strength=1.0):
    image_f = image.astype(np.float32)
    sharpened = image_f.copy()
    for sigma in sigmas:
        blur = cv2.GaussianBlur(image_f, (0, 0), sigma)
        detail = image_f - blur
        sharpened += strength * detail
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def adaptive_unsharp_masking(image, kernel_size=5, strength=1.8):
    """Adaptive unsharp masking, adjusting parameters based on local contrast"""

    result = np.zeros_like(image)
    mean, stddev = cv2.meanStdDev(image)
    for i in range(3):
        blurred = cv2.GaussianBlur(image[:,:,i], (kernel_size, kernel_size), 0)
        detail = image[:,:,i].astype(np.float32) - blurred.astype(np.float32)
        local_strength = strength * (0.5 + stddev[i]/128 + 0.5 * (1 - mean[i]/255))
        result[:,:,i] = np.clip(image[:,:,i] + local_strength * detail, 0, 255).astype(np.uint8)
    return result

def motion_deblur(image, angle, strength):
    """Construct motion blur kernel and perform Wiener deconvolution"""
    
    def motion_kernel(length, angle):
        kernel = np.zeros((length, length), dtype=np.float32)
        cv2.line(kernel, (0, length // 2), (length - 1, length // 2), 1, thickness=1)
        kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1.0),
                                (length, length))
        kernel /= np.sum(kernel)
        return kernel
    
    kernel = motion_kernel(strength, angle)
    result = np.zeros_like(image)
    kernel = motion_kernel(strength, angle)
    for i in range(3):
        result[:, :, i] = wiener_deconv(image[:, :, i], kernel, 0.01)
    return result

def wiener_deconv(image, kernel, noise_power):
    
    def _deconv_channel(img, kernel):
        img = img.astype(np.float32) / 255.0
        h, w = img.shape
        kh, kw = kernel.shape
        psf = np.zeros((h, w), dtype=np.float32)
        psf[:kh, :kw] = kernel
        psf = np.roll(psf, -kh // 2, axis=0)
        psf = np.roll(psf, -kw // 2, axis=1)
        img_fft = np.fft.fft2(img)
        psf_fft = np.fft.fft2(psf)
        psf_fft_conj = np.conj(psf_fft)
        result_fft = img_fft * psf_fft_conj / (np.abs(psf_fft)**2 + noise_power)
        result = np.real(np.fft.ifft2(result_fft))
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        output = cv2.GaussianBlur(result, (3, 3), 0.5)
        output = cv2.edgePreservingFilter(output, flags=1, sigma_s=60, sigma_r=0.3)
        return output
    
    if image.ndim == 3 and image.shape[2] == 3:
        return np.dstack([_deconv_channel(image[:, :, i], kernel) for i in range(3)])
    else:
        return _deconv_channel(image, kernel)

def fake_hdr_enhance(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    gamma_corrected = np.power(l_eq / 255.0, 1.2) * 255
    l_final = np.clip(gamma_corrected, 0, 255).astype(np.uint8)
    merged = cv2.merge([l_final, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def enhance_image_quality(image, methods=None):
    """
    综合图像增强方法：可选 Unsharp、Adaptive、Wiener、Motion Deblur 等。
    """
    if methods is None:
        methods = ['all']

    results = {}
    base_image = image.copy()
    results['original'] = base_image

    # -------------------- Unsharp Masking --------------------
    if 'unsharp' in methods or 'all' in methods:
        results['unsharp'] = strong_unsharp_masking(
            base_image, sigma=1, strength=2, iterations=2,
            edge_boost=True, edge_weight=0.3
        )

    # -------------------- Adaptive Unsharp --------------------
    if 'adaptive' in methods or 'all' in methods:
        results['adaptive'] = adaptive_unsharp_masking(
            base_image, kernel_size=5, strength=2.0
        )

    # -------------------- multiscale Unsharp --------------------    
    if 'multiscale' in methods or 'all' in methods:
        results['multiscale'] = multiscale_unsharp(
            base_image, sigmas=[1.0, 2.0, 3.0], strength=1.2
        )

    if 'fake_hdr' in methods or 'all' in methods:
        results['fake_hdr'] = fake_hdr_enhance(image)

    # -------------------- Wiener Deconvolution --------------------
    if 'wiener_deconv' in methods or 'all' in methods:
        preprocessed = clahe_then_sharpen(base_image)
        preprocessed = laplacian_enhance(preprocessed)
        kernel = np.ones((5, 5), dtype=np.float32)
        kernel /= kernel.sum()
        results['wiener_deconv'] = wiener_deconv(
            preprocessed, kernel=kernel, noise_power=0.08
        )

    # -------------------- Motion Deblur --------------------
    def smart_motion_deblur(image, angle, strength):
        deblur = motion_deblur(image, angle=angle, strength=strength)
        final = light_sharpen(deblur)
        return final

    if 'motion_deblur' in methods or 'all' in methods:
        results['motion_30'] = smart_motion_deblur(base_image, angle=30, strength=15)
        results['motion_60'] = smart_motion_deblur(base_image, angle=60, strength=15)
        results['motion_90'] = smart_motion_deblur(base_image, angle=90, strength=3)

    return results

def process_image(image, output_folder=None):
    """Process a single image and display/save the result"""
    if output_folder is not None and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if image is None:
        print(f"Unable to read image:{image_path}")
        return
    results = enhance_image_quality(image)
    for method, result in results.items():
        if method == 'original':
            continue
        # save results
        if output_folder is not None:
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(output_folder, f"{name}_{method}{ext}")
            cv2.imwrite(output_path, result)
            print(f"Saved: {output_path}")
    scores = {}
    for method, result in results.items():
        if method == 'original':
            continue
        score = evaluate_quality(result)
        scores[method] = score
        print(f"{method} clarity score:  {score:.3f}")

    best_method = max(scores, key=scores.get)
    best_result = results[best_method]
    print(f"\nThe best enhancement methods are:{best_method}(Highest clarity score)")

    show_comparison(image, best_result, "original_image", f"{best_method}")

    if output_folder is not None:
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_folder, f"{name}_{best_method}{ext}")
        cv2.imwrite(output_path, best_result)
        print(f"Best images saved:{output_path}")


if __name__ == "__main__":
    image_path =  "./img/deblur-4.jpg"
    output_folder = "enhanced_results"
    image = cv2.imread(image_path)
    width = None
    height = None
    image = resize_image(image, width, height)
    process_image(image, output_folder)