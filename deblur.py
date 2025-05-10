import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
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

def evaluate_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    score = laplacian.var()
    return score

def show_comparison(original, enhanced, title1="Original", title2="Enhanced"):
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

def strong_unsharp_masking(image, sigma=1.0, strength=2.0, iterations=1, edge_boost=False, edge_weight=0.3):
    result = image.copy()
    for _ in range(iterations):
        blurred = cv2.GaussianBlur(result, (0, 0), sigma)
        result = cv2.addWeighted(result, 1.0 + strength, blurred, -strength, 0)
        result = np.clip(result, 0, 255).astype(np.uint8)
    if edge_boost:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        result = cv2.addWeighted(result, 1.0, edges_colored, edge_weight, 0)
    return np.clip(result, 0, 255).astype(np.uint8)

def edge_preserve_then_sharpen(image):
    smooth = cv2.edgePreservingFilter(image, flags=1, sigma_s=60, sigma_r=0.4)
    sharpened = cv2.addWeighted(image, 1.5, smooth, -0.5, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def fake_hdr_enhance(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    gamma_corrected = np.power(l_eq / 255.0, 1.2) * 255
    l_final = np.clip(gamma_corrected, 0, 255).astype(np.uint8)
    merged = cv2.merge([l_final, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def denoise_clahe_sharpen(image):
    denoised = cv2.medianBlur(image, 3)
    clahe_img = clahe_then_sharpen(denoised)
    return strong_unsharp_masking(clahe_img, sigma=1, strength=1.8, iterations=2, edge_boost=False)

def motion_kernel(length, angle):
    kernel = np.zeros((length, length), dtype=np.float32)
    cv2.line(kernel, (0, length // 2), (length - 1, length // 2), 1, thickness=1)
    M = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (length, length))
    return kernel / np.sum(kernel)

def wiener_deconv(image, kernel, noise_power=0.01):
    def deconv_channel(channel, kernel):
        img = channel.astype(np.float32) / 255.0
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
        return np.clip(result * 255, 0, 255).astype(np.uint8)
    return np.dstack([deconv_channel(image[:, :, i], kernel) for i in range(3)])

def motion_deblur(image, angle=30, strength=15):
    kernel = motion_kernel(strength, angle)
    return wiener_deconv(image, kernel, noise_power=0.01)

def motion_deblur_sweep(image, angles=[0, 30, 60, 90], lengths=[13, 17, 21], noise_power=0.01):
    def wiener_deconv_gray(image, kernel):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        h, w = gray.shape
        kh, kw = kernel.shape
        psf = np.zeros((h, w), dtype=np.float32)
        psf[:kh, :kw] = kernel
        psf = np.roll(psf, -kh // 2, axis=0)
        psf = np.roll(psf, -kw // 2, axis=1)
        img_fft = np.fft.fft2(gray)
        psf_fft = np.fft.fft2(psf)
        psf_fft_conj = np.conj(psf_fft)
        result_fft = img_fft * psf_fft_conj / (np.abs(psf_fft)**2 + noise_power)
        result = np.real(np.fft.ifft2(result_fft))
        result = np.clip(result * 255, 0, 255).astype(np.uint8)

        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        yuv[:, :, 0] = result
        return cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR)

    results = []
    for angle in angles:
        for length in lengths:
            kernel = motion_kernel(length, angle)
            deblurred = wiener_deconv_gray(image, kernel)
            score = evaluate_quality(deblurred)
            results.append((score, angle, length, deblurred))

    results.sort(reverse=True, key=lambda x: x[0])
    top_results = results[:3]
    print("\n[Motion Deblur Sweep - Top 3 Results]")
    for i, (score, angle, length, _) in enumerate(top_results):
        print(f"#{i+1}: Angle={angle}, Length={length}, Clarity Score={score:.2f}")

    return {f"motion_best{i+1}_angle{angle}_len{length}": img for i, (score, angle, length, img) in enumerate(top_results)}


def enhance_image_quality(image):
    results = {'original': image}
    base_image = image.copy()
    mean_brightness = cv2.mean(cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY))[0]
    sharpness = evaluate_quality(base_image)

    if mean_brightness < 80:
        results['fake_hdr'] = fake_hdr_enhance(base_image)
    if sharpness < 50:
        results['denoise_clahe'] = denoise_clahe_sharpen(base_image)

    results['unsharp'] = strong_unsharp_masking(base_image)
    results['edge_preserve'] = edge_preserve_then_sharpen(base_image)
    results.update(motion_deblur_sweep(base_image))

    return results

# def process_image(image, output_folder=None, image_name="processed_image"):
#     results = enhance_image_quality(image)
#     scores = {}
#     for method, result in results.items():
#         if method != 'original':
#             score = evaluate_quality(result)
#             scores[method] = score
#             print(f"{method}: clarity score = {score:.3f}")
#             if output_folder:
#                 os.makedirs(output_folder, exist_ok=True)
#                 name = os.path.splitext(os.path.basename(image_name))[0]
#                 cv2.imwrite(os.path.join(output_folder, f"{name}_{method}.png"), result)
#     best = max(scores, key=scores.get)
#     print(f" Best method: {best}")
#     show_comparison(image, results[best], "Original", f"Best: {best}")

# def process_image_before(image, output_folder=None, width=None, height=None):
#     if image is None:
#         print(f"❌ Failed to read: {image}")
#         return
#     image = resize_image(image, width, height)
#     process_image(image, output_folder)

def deblur_process_image(image, output_folder=None, image_name="processed_image"):
    results = enhance_image_quality(image)
    scores = {}

    for method, result in results.items():
        if method != 'original':
            score = evaluate_quality(result)
            scores[method] = score
            print(f"{method}: clarity score = {score:.3f}")

            if output_folder:
                os.makedirs(output_folder, exist_ok=True)
                name = os.path.splitext(image_name)[0]
                output_path = os.path.join(output_folder, f"{name}_{method}.png")
                cv2.imwrite(output_path, result)

    best = max(scores, key=scores.get)
    print(f"✅ Best method: {best}")

    show_comparison(image, results[best], "Original", f"Best: {best}")
    return results[best]  