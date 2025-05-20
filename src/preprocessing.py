import cv2
import numpy as np

def remove_glare(image):
    """Remove glare from an image using HSV thresholding and inpainting."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    _, glare_mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, saturation_mask = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY_INV)
    glare_mask = cv2.bitwise_and(glare_mask, saturation_mask)
    kernel = np.ones((3, 3), np.uint8)
    glare_mask = cv2.dilate(glare_mask, kernel, iterations=1)
    glare_mask = cv2.erode(glare_mask, kernel, iterations=1)
    image_no_glare = cv2.inpaint(image, glare_mask, 5, cv2.INPAINT_TELEA)
    mask_area = np.sum(glare_mask == 255)
    image_area = image.shape[0] * image.shape[1]
    alpha = 0.7 if mask_area / image_area <= 0.3 else 0.9
    image_no_glare = cv2.addWeighted(image, alpha, image_no_glare, 1 - alpha, 0)
    return image_no_glare, glare_mask

def remove_hair(image):
    """Remove hair from an image using morphological operations and inpainting."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(image, hair_mask, 3, cv2.INPAINT_TELEA)
    return inpainted, hair_mask

def process_image(img_path):
    """Process an image by removing hair and glare."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load {img_path}")
        return None
    img_hair_removed, hair_mask = remove_hair(img)
    img_fully_processed, glare_mask = remove_glare(img_hair_removed)
    return {
        'original': img,
        'hair_removed': img_hair_removed,
        'glare_removed': img_fully_processed,
        'hair_mask': hair_mask,
        'glare_mask': glare_mask
    }