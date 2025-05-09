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
def main(image_path):
    image = Image.open(image_path)
    brightness = estimate_brightness(image)
    
    # Assuming atmospheric light is the image's maximum
    airlight = np.max(np.array(image.convert('RGB')), axis=(0, 1))
    
    # Estimated transmittance
    transmission = estimate_transmission(image, brightness, alpha=0.7)
    
    # Recovery of de-fogged images using atmospheric scattering models
    dehazed_image = dehaze(image, transmission, airlight)
    result_img = Image.fromarray(dehazed_image)
    
    result_img.show()
    result_img.save('/Users/ruoyan/Desktop/Master/Year1-Sem1/COMP5405-Digital Media Computing/Project/output.jpg')

# Run the main function
if __name__ == "__main__":
    main('/Users/ruoyan/Desktop/Master/Year1-Sem1/COMP5405-Digital Media Computing/Project/image2.png')
