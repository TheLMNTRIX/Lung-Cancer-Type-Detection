import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
import cv2
import os
import numpy as np

# Define input and output paths for data
windows_input_path = r'C:\Users\rohit\OneDrive\Desktop\Data\Original'  # Path to original dataset
windows_output_path = r'C:\Users\rohit\OneDrive\Desktop\Data\PreProcessed2'  # Path for preprocessed images

# Convert Windows paths to WSL-compatible paths (for cross-platform compatibility)
input_dir = f"/mnt/{windows_input_path[0].lower()}{windows_input_path[2:].replace('\\', '/')}"
output_dir = f"/mnt/{windows_output_path[0].lower()}{windows_output_path[2:].replace('\\', '/')}"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Set the target size for image resizing
IMG_SIZE = (224, 224)

# Helper function to preprocess the image without augmentation
def preprocess_image_without_augmentation(image_path, img_size=IMG_SIZE):
    # Load image as a PIL object and convert to NumPy array
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0  # Scale to [0, 1] for consistency
    return img_array

# Data augmentation function for medical CT scan images
def augment_image(img_array):
    # Convert the image array back to uint8 format
    img_array = (img_array * 255).astype(np.uint8)

    # Random rotation within a range of +/- 15 degrees
    angle = np.random.uniform(-8, 8)
    (h, w) = img_array.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    rotated_img = cv2.warpAffine(img_array, rotation_matrix, (w, h))

    # Random brightness adjustment between 80% to 120%
    brightness_factor = np.random.uniform(0.8, 1.2)
    augmented_img = cv2.convertScaleAbs(rotated_img, alpha=brightness_factor, beta=0)

    # Resize and rescale the augmented image to [0, 1] for TensorFlow compatibility
    augmented_img_resized = cv2.resize(augmented_img, IMG_SIZE) / 255.0
    return augmented_img_resized

# Function to preprocess, augment, and save both original and augmented images
def save_preprocessed_images(input_dir, output_dir, img_size=IMG_SIZE):
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)  # Directory for each class
        class_save_dir = os.path.join(output_dir, class_name)  # Output directory for each class
        os.makedirs(class_save_dir, exist_ok=True)

        # Process each image in the class directory
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            
            # 1. Preprocess without augmentation and save
            preprocessed_img = preprocess_image_without_augmentation(img_path, img_size)
            save_path_preprocessed = os.path.join(class_save_dir, img_name)
            cv2.imwrite(save_path_preprocessed, (preprocessed_img * 255).astype(np.uint8))  # Convert to [0, 255] range for saving
            
            # 2. Apply augmentation and save as a new file
            augmented_img = augment_image(preprocessed_img)
            save_path_augmented = os.path.join(class_save_dir, f"aug_{img_name}")  # Save with "aug_" prefix to differentiate
            cv2.imwrite(save_path_augmented, (augmented_img * 255).astype(np.uint8))  # Convert to [0, 255] range for saving

# Ensure GPU is being used if available
print("Using GPU: ", "Yes" if tf.config.list_physical_devices('GPU') else "No")

# Loop over data splits (train, valid, test), preprocess, and save images
for split in ['train', 'valid', 'test']:
    split_input_dir = os.path.join(input_dir, split)  # Input directory for the split
    split_output_dir = os.path.join(output_dir, split)  # Output directory for the split
    os.makedirs(split_output_dir, exist_ok=True)
    save_preprocessed_images(split_input_dir, split_output_dir)

print("Preprocessing and augmentation complete, original and augmented images saved.")
