import numpy as np
import cv2

# UIQM SALAH 
# UCIQE SALAH
# def entropy(img):
#     hist, _ = np.histogram(img, bins=256, range=(0,255), density=True)
#     hist = hist[np.nonzero(hist)]
#     ent = -np.sum(hist * np.log2(hist))
#     return ent

# def colorfulness(img):
#     R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
#     rg = np.abs(R - G)
#     yb = np.abs(0.5 * (R + G) - B)
#     std_rg = np.std(rg)
#     std_yb = np.std(yb)
#     mean_rg = np.mean(rg)
#     mean_yb = np.mean(yb)
#     cf = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
#     return cf

# def sharpness(img_gray):
#     from scipy.ndimage import sobel
#     sobel_x = sobel(img_gray, axis=0)
#     sobel_y = sobel(img_gray, axis=1)
#     grad = np.sqrt(sobel_x**2 + sobel_y**2)
#     sharp = np.mean(grad)
#     return sharp

# def uiqm(img):
#     if img.dtype != np.uint8:
#         img = (img * 255).astype(np.uint8)
#     img_gray = np.mean(img, axis=2)
#     c = colorfulness(img)
#     s = sharpness(img_gray)
#     e = entropy(img_gray)
#     return 0.0282*c + 3.5521*s + 0.0484*e

# Usage Example:
# import cv2
# img = cv2.imread('underwater.jpg')
# score = uiqm(img)
# print(score)



# def uciqe(img):
#     # Convert RGB to Lab color space
#     img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     L = img_lab[:,:,0].astype(np.float32)
#     a = img_lab[:,:,1].astype(np.float32)
#     b = img_lab[:,:,2].astype(np.float32)
    
#     # Chromaticity (ch) calculation
#     chroma = np.sqrt(a**2 + b**2)
#     std_chroma = np.std(chroma)
    
#     # Contrast of luminance channel
#     L_lin = L.flatten()
#     contrast = np.max(L_lin) - np.min(L_lin)
    
#     # Mean of saturation
#     R, G, B = img[:,:,2], img[:,:,1], img[:,:,0]  # OpenCV uses BGR
#     max_rgb = np.maximum(np.maximum(R, G), B)
#     min_rgb = np.minimum(np.minimum(R, G), B)
#     sum_rgb = R + G + B + 1e-6
#     saturation = 1 - 3 * min_rgb / sum_rgb
#     mean_saturation = np.mean(saturation)
    
#     # UCIQE formula
#     # α, β, γ values from the original paper: α=0.4680, β=0.2745, γ=0.2576
#     uciqe_score = 0.4680 * std_chroma + 0.2745 * contrast + 0.2576 * mean_saturation
#     return uciqe_score

# Usage:
# import cv2
# img = cv2.imread('underwater.jpg')
# score = uciqe(img)
# print(score)


import numpy as np
from scipy.ndimage import uniform_filter

def pcqi(img, win_size=7):
    # Convert image to grayscale if not already
    if len(img.shape) == 3:
        img_gray = np.mean(img, axis=2)
    else:
        img_gray = img.copy()
    img_gray = img_gray.astype(np.float64)

    # Local mean and local contrast
    local_mean = uniform_filter(img_gray, win_size)
    local_sq_mean = uniform_filter(img_gray**2, win_size)
    local_std = np.sqrt(local_sq_mean - local_mean**2 + 1e-8)

    # Global mean and standard deviation
    global_mean = np.mean(img_gray)
    global_std = np.std(img_gray)

    # PCQI formula: combines patch structure index
    pcqi_index = np.mean((local_std/global_std) * (local_mean/global_mean))
    return pcqi_index

# Usage:
# import cv2
# img = cv2.imread('underwater.jpg')
# score = pcqi(img)
# print(score)


# Install the package first:
# pip install pypiqe


# img = cv2.imread('image.jpg')  # Can be grayscale or RGB
# score, activityMask, noticeableArtifactMask, noiseMask = piqe(img)
# print('PIQE score:', score)
from pypiqe import piqe
def get_piqe(img):
    score, activityMask, noticeableArtifactMask, noiseMask = piqe(img)
    return score

def dark_channel(img, win_size=15):
    # Minimum over RGB channels
    min_channel = np.min(img, axis=2)
    # Apply minimum filter (dark channel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win_size, win_size))
    dark = cv2.erode(min_channel, kernel)
    return dark

def local_contrast(img_gray, win_size=15):
    mean = cv2.blur(img_gray, (win_size, win_size))
    std = np.sqrt(cv2.blur(img_gray**2, (win_size, win_size)) - mean**2)
    return std

def fade(img):
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    # Compute dark channel
    dark = dark_channel(img, win_size=15)
    # Compute local contrast (from the gray image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    contrast = local_contrast(img_gray, win_size=15)
    # FADE formula: combine features (weights used in literature)
    fade_score = np.mean(dark) - np.mean(contrast)
    return fade_score

# Usage example:
# img = cv2.imread('foggy_image.jpg')
# score = fade(img)
# print('FADE score:', score)

import numpy as np
import cv2

def average_gradient(img):
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Calculate x and y gradients with Sobel operator
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Return average gradient magnitude
    return np.mean(magnitude)

# Example usage:
# import cv2
# img = cv2.imread('image.jpg')
# ag_score = average_gradient(img)
# print('Average Gradient:', ag_score)
