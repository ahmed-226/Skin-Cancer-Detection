import cv2
import numpy as np
import mahotas
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from pywt import dwt2

def extract_color_histogram(image, mask=None, bins=64):
    """Extract color histogram (192 features)."""
    if mask is not None:
        masked_image = image.copy()
        masked_image[mask == 0] = 0
    else:
        masked_image = image
    hist_r = np.histogram(masked_image[:, :, 0].ravel(), bins=bins, range=(0, 256))[0]
    hist_g = np.histogram(masked_image[:, :, 1].ravel(), bins=bins, range=(0, 256))[0]
    hist_b = np.histogram(masked_image[:, :, 2].ravel(), bins=bins, range=(0, 256))[0]
    hist_r = hist_r / (hist_r.sum() + 1e-10)
    hist_g = hist_g / (hist_g.sum() + 1e-10)
    hist_b = hist_b / (hist_b.sum() + 1e-10)
    return np.concatenate([hist_r, hist_g, hist_b])

def extract_color_moments(image, mask=None):
    """Extract color moments in RGB (mean, variance, skewness, kurtosis; 12 features)."""
    if mask is not None:
        masked_image = image.copy()
        masked_image[mask == 0] = 0
    else:
        masked_image = image
    moments = []
    for channel in range(3):
        pixels = masked_image[:, :, channel].ravel()
        pixels = pixels[pixels != 0] if mask is not None else pixels
        if len(pixels) == 0:
            moments.extend([0, 0, 0, 0])
            continue
        mean = np.mean(pixels)
        variance = np.var(pixels)
        skewness = skew(pixels, bias=False) if variance > 0 else 0
        kurt = kurtosis(pixels, bias=False) if variance > 0 else 0
        moments.extend([mean, variance, skewness, kurt])
    return np.array(moments)

def extract_hsv_stats(image, mask=None):
    """Extract mean and std of HSV channels (6 features)."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    if mask is not None:
        masked_hsv = hsv_image.copy()
        masked_hsv[mask == 0] = 0
    else:
        masked_hsv = hsv_image
    stats = []
    for channel in range(3):
        pixels = masked_hsv[:, :, channel].ravel()
        pixels = pixels[pixels != 0] if mask is not None else pixels
        if len(pixels) == 0:
            stats.extend([0, 0])
            continue
        mean = np.mean(pixels)
        std = np.std(pixels)
        stats.extend([mean, std])
    return np.array(stats)

def extract_lab_stats(image, mask=None):
    """Extract mean and std of LAB channels (6 features)."""
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    if mask is not None:
        masked_lab = lab_image.copy()
        masked_lab[mask == 0] = 0
    else:
        masked_lab = lab_image
    stats = []
    for channel in range(3):
        pixels = masked_lab[:, :, channel].ravel()
        pixels = pixels[pixels != 0] if mask is not None else pixels
        if len(pixels) == 0:
            stats.extend([0, 0])
            continue
        mean = np.mean(pixels)
        std = np.std(pixels)
        stats.extend([mean, std])
    return np.array(stats)

def extract_haralick_and_stats(image, mask=None, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """Extract Haralick features and intensity stats (mean, std, entropy, kurtosis; 17 features)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if mask is not None:
        gray[mask == 0] = 0
    pixels = gray.ravel()
    pixels = pixels[pixels != 0] if mask is not None else pixels
    if len(pixels) == 0:
        stats = [0, 0, 0, 0]
    else:
        mean = np.mean(pixels)
        std = np.std(pixels)
        entropy = -np.sum([(p/pixels.sum()) * np.log2(p/pixels.sum() + 1e-10) 
                          for p in np.histogram(pixels, bins=256)[0] if p > 0])
        kurt = kurtosis(pixels, bias=False) if std > 0 else 0
        stats = [mean, std, entropy, kurt]
    if mask is not None:
        gray_for_glcm = gray.copy()
        gray_for_glcm[mask == 0] = 0
    else:
        gray_for_glcm = gray
    gray_for_glcm = gray_for_glcm.astype(np.uint8)
    glcm = graycomatrix(gray_for_glcm, distances=distances, angles=angles, 
                        levels=256, symmetric=True, normed=True)
    haralick_features = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in properties:
        prop_values = graycoprops(glcm, prop).mean(axis=1).ravel()
        haralick_features.extend(prop_values)
    haralick_mahotas = mahotas.features.haralick(gray_for_glcm, return_mean=True)
    extra_features = haralick_mahotas[[0, 1, 2, 3, 4, 8]]
    return np.concatenate([haralick_features, extra_features, stats])

def extract_wavelet_features(image, mask=None, wavelet='db1'):
    """Extract wavelet transform features (mean and std of coefficients; 6 features)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if mask is not None:
        gray[mask == 0] = 0
    coeffs = dwt2(gray, wavelet)
    cA, (cH, cV, cD) = coeffs
    features = []
    for coeff in [cH, cV, cD]:
        pixels = coeff.ravel()
        pixels = pixels[pixels != 0] if mask is not None else pixels
        if len(pixels) == 0:
            features.extend([0, 0])
        else:
            mean = np.mean(pixels)
            std = np.std(pixels)
            features.extend([mean, std])
    return np.array(features)

def extract_lbp(image, mask=None, radius=3, n_points=24):
    """Extract LBP histogram (26 features)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if mask is not None:
        gray[mask == 0] = 0
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), 
                          range=(0, n_points + 2), density=True)
    return hist

def extract_hu_moments(image, mask=None):
    """Extract Hu Moments (7 features)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if mask is not None:
        gray[mask == 0] = 0
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).ravel()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    return hu_moments

def extract_diameter(image, mask=None):
    """Extract lesion diameter (1 feature)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if mask is None:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        diameter = 0
    else:
        contour = max(contours, key=cv2.contourArea)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            diameter = max(ellipse[1])
        else:
            area = cv2.contourArea(contour)
            diameter = 2 * np.sqrt(area / np.pi)
    return np.array([diameter])

def extract_circularity(image, mask=None):
    """Extract Circularity (1 feature)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if mask is None:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        circularity = 0
    else:
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-10)
    return np.array([circularity])

def extract_asymmetry_and_border(image, mask=None):
    """Extract asymmetry and border irregularity (2 features)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if mask is None:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = mask.astype(np.uint8)
    flipped = cv2.flip(mask, 1)
    intersection = cv2.bitwise_and(mask, flipped)
    union = cv2.bitwise_or(mask, flipped)
    asymmetry_index = 1 - (np.sum(intersection) / (np.sum(union) + 1e-10))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        border_irregularity = 0
    else:
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        border_irregularity = perimeter ** 2 / (4 * np.pi * area + 1e-10)
    return np.array([asymmetry_index, border_irregularity])

def normalize_features(features):
    """Normalize features to [0, 1] range."""
    if features.size == 0:
        return features
    min_val = np.min(features)
    max_val = np.max(features)
    if max_val == min_val:
        return np.zeros_like(features)
    return (features - min_val) / (max_val - min_val)

def extract_all_features(image, mask=None):
    """Extract all features (color, texture, shape) and normalize."""
    color_hist = extract_color_histogram(image, mask, bins=64)  # 192 features
    color_moments = extract_color_moments(image, mask)  # 12 features
    hsv_stats = extract_hsv_stats(image, mask)  # 6 features
    lab_stats = extract_lab_stats(image, mask)  # 6 features
    haralick_stats = extract_haralick_and_stats(image, mask)  # 17 features
    wavelet_features = extract_wavelet_features(image, mask)  # 6 features
    lbp_hist = extract_lbp(image, mask, radius=3, n_points=24)  # 26 features
    hu_moments = extract_hu_moments(image, mask)  # 7 features
    diameter = extract_diameter(image, mask)  # 1 feature
    circularity = extract_circularity(image, mask)  # 1 feature
    asym_border = extract_asymmetry_and_border(image, mask)  # 2 features
    features = np.concatenate([
        color_hist,
        color_moments,
        hsv_stats,
        lab_stats,
        haralick_stats,
        wavelet_features,
        lbp_hist,
        hu_moments,
        diameter,
        circularity,
        asym_border
    ])
    features = normalize_features(features)
    return features