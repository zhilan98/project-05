from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize

def piecewise_affine_warp(image_path, mode='sine', strength=10, grid_size=8, show=True):
    """
    Apply piecewise affine warp to an image with different deformation modes.

    Parameters:
        img      : Input image (H x W x C)
        mode     : Type of deformation ('sine', 'wave_x', 'zoom')
        strength : Degree of deformation
        grid_size: Number of control points per row/col
        show     : Whether to show the result with matplotlib

    Returns:
        warped image (same shape as input)
    """
    img = cv2.imread(image_path)
    img = transform.resize(img, (img.shape[0], img.shape[1]), anti_aliasing=True)

    rows, cols = grid_size, grid_size
    src_cols = np.linspace(0, img.shape[1], cols)
    src_rows = np.linspace(0, img.shape[0], rows)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    dst = src.copy()

    if mode == 'sine':
        dst[:, 1] += strength * np.sin(np.linspace(0, 2 * np.pi, len(dst)))
    elif mode == 'wave_x':
        dst[:, 0] += strength * np.sin(np.linspace(0, 2 * np.pi, len(dst)))
    elif mode == 'zoom':
        center = np.mean(src, axis=0)
        direction = src - center
        dst = src - strength * direction / np.linalg.norm(direction, axis=1)[:, np.newaxis]
    else:
        raise ValueError("Unsupported mode. Choose from 'sine', 'wave_x', 'zoom'.")

    
    tform = transform.PiecewiseAffineTransform()
    tform.estimate(src, dst)
    warped = transform.warp(img, tform)

    if show:
        visualize_affine_result(img, warped, mode)

    return warped

def visualize_affine_result(original, warped, mode_name=''): 
    """ Helper function to visualize original and warped images side by side. """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(warped)
    plt.title(f"Warped ({mode_name})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# # Load image with OpenCV
# img = cv2.imread('example_img1.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
# img = resize(img, (256, 256))  # Resize to standard size

# # Apply different warp modes
# piecewise_affine_warp(img, mode='sine')

# piecewise_affine_warp(img, mode='wave_x', strength=40, grid_size=30)

# piecewise_affine_warp(img, mode='zoom', strength=40, grid_size=30)
