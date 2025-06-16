import cv2
import os
import numpy as np

image_dir = '/home/htic/Desktop/raisa/astrocytes/astrocyte-detection/test-media/final-test'
mask_dir= '/home/htic/Desktop/raisa/astrocytes/astrocyte-detection/test-media/final-test-mask'
output_dir = '/home/htic/Desktop/raisa/astrocytes/astrocyte-detection/SimCLR_U-net/mask-view'

def apply_translucent_mask(image_dir, mask_dir, output_dir, alpha=0.5):
    """
    Applies a translucent red binary mask on original images and saves the result.

    Args:
        image_dir (str): Directory containing original JPEG images.
        mask_dir (str): Directory containing TIFF binary mask images.
        output_dir (str): Directory to save the output images with overlay.
        alpha (float): Transparency factor for the mask (0.0 for fully transparent,
                       1.0 for fully opaque). Default is 0.5.
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {os.path.abspath(output_dir)}")

    for i in os.listdir(image_dir):
        if i.endswith('.jpeg'): # Also consider .jpg extension
            image_filename = i
            mask_filename = i.replace('.jpeg', '_ensembled_mask.png')

            image_path = os.path.join(image_dir, image_filename)
            mask_path = os.path.join(mask_dir, mask_filename)
            output_path = os.path.join(output_dir, image_filename.replace('.jpeg', 'view-mask.png'))

            # Check if mask file exists for the current image
            if not os.path.exists(mask_path):
                print(f"Warning: Mask not found for {image_filename}. Skipping.")
                continue

            # Load the image and mask
            # cv2.IMREAD_COLOR loads a 3-channel image. If it's grayscale, it will convert.
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            # cv2.IMREAD_GRAYSCALE loads a single-channel mask.
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Error: Could not load image {image_path}. Skipping.")
                continue
            if mask is None:
                print(f"Error: Could not load mask {mask_path}. Skipping.")
                continue

            # Ensure image and mask have the same dimensions
            if image.shape[:2] != mask.shape[:2]:
                print(f"Warning: Image {image_filename} and mask dimensions mismatch. Resizing mask.")
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Convert mask to binary (0 or 255)
            # Thresholding ensures the mask is strictly binary (0 or 255)
            _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

            # Create a 3-channel color mask (red: BGR=(0, 0, 255))
            # The red mask will only have color where binary_mask is 255
            red_mask = np.zeros_like(image, dtype=np.uint8)
            red_mask[binary_mask == 255] = [0, 0, 255] # Set to Red (B=0, G=0, R=255)

            # Overlay the red mask onto the original image
            # The formula is: output = image * (1 - alpha) + red_mask * alpha
            # This blends the two images with the specified transparency.
            overlay_image = cv2.addWeighted(image, 1 - alpha, red_mask, alpha, 0)

            # Save the combined image
            cv2.imwrite(output_path, overlay_image)
            print(f"Processed and saved: {output_path}")
            
apply_translucent_mask(image_dir, mask_dir, output_dir, alpha=0.5)