import cv2
import matplotlib.pyplot as plt

def visualize_processed_image(processed_img):
    """Visualize preprocessing steps (original, hair mask, hair removed, glare mask, fully processed)."""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 5, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(processed_img['original'], cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 5, 2)
    plt.title("Hair Mask")
    plt.imshow(processed_img['hair_mask'], cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 5, 3)
    plt.title("Hair Removed")
    plt.imshow(cv2.cvtColor(processed_img['hair_removed'], cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 5, 4)
    plt.title("Glare Mask")
    plt.imshow(processed_img['glare_mask'], cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 5, 5)
    plt.title("All Processed")
    plt.imshow(cv2.cvtColor(processed_img['glare_removed'], cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_fully_processed(processed_img):
    """Compare original and fully processed image."""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(processed_img['original'], cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Fully Processed (Hair & Glare Removed)")
    plt.imshow(cv2.cvtColor(processed_img['glare_removed'], cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()