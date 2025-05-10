import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# Estimate the brightness of an image (convert image to grayscale)
def estimate_brightness(image):
    image_array = np.array(image.convert('RGB'))
    brightness = np.mean(image_array, axis=2)  # Calculate the brightness of each pixel (RGB average)
    return brightness

# Estimated transmittance
def estimate_transmission(image, brightness, alpha=0.7):
    # Use the overall brightness of the image to estimate transmittance
    transmission = 1 - alpha * (brightness / np.max(brightness))  
    transmission = np.clip(transmission, 0.1, 1)  
    return transmission

# Image recovery using atmospheric scattering model
def dehaze(image, transmission, airlight, size=15):
    image_array = np.array(image.convert('RGB')).astype(np.float32)
    
    # Smoothing of transmittance using Gaussian filters
    smoothed_transmission = gaussian_filter(transmission, sigma=size)
    
    # Recover the image
    result = (image_array - airlight) / smoothed_transmission[:, :, np.newaxis] + airlight
    return np.clip(result, 0, 255).astype(np.uint8)

# main function
def dehaze_image(image_path):
    image = Image.open(image_path)
    brightness = estimate_brightness(image)
    airlight = np.max(np.array(image.convert('RGB')), axis=(0, 1))
    transmission = estimate_transmission(image, brightness)
    dehazed_array = dehaze(image, transmission, airlight)
    return Image.fromarray(dehazed_array)
