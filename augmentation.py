import os
import numpy as np
import random

# Augmentation settings
AUGMENTATION_MULTIPLIER = 2.5  
OFFSET_RANGE = (-15, 15)  # Bounding box offset range (pixels)
IMG_WIDTH = 1920  # Adjust based on actual dataset resolution
IMG_HEIGHT = 1080  # Adjust based on actual dataset resolution

# Paths for dataset
INPUT_FOLDER = r"dataset_output\extracted_frames_normal_behavior\cleaned_keypoints"
OUTPUT_FOLDER = r"dataset_output\extracted_frames_normal_behavior\augmented_keypoints"

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def load_keypoints(file_path):
    """
    Load keypoints and bounding box from a file.

    Args:
        file_path (str): Path to the keypoint file.

    Returns:
        tuple: Bounding box (list) and keypoints (numpy array).
    """
    with open(file_path, 'r') as f:
        values = list(map(float, f.readline().strip().split(',')))

    # Bounding box is first 4 values
    bounding_box = values[:4]  # (x1, y1, x2, y2)

    # Remaining values are keypoints
    keypoints = np.array(values[4:]).reshape(17, 3)  # (17 keypoints, x, y, confidence)

    return bounding_box, keypoints

def save_augmented_keypoints(bounding_box, keypoints, output_folder, filename):
    """
    Save augmented bounding box and keypoints to a new file.

    Args:
        bounding_box (list): Augmented bounding box.
        keypoints (np.array): Original keypoints.
        output_folder (str): Directory to save the file.
        filename (str): Name of the new file.
    """
    output_path = os.path.join(output_folder, filename)
    with open(output_path, 'w') as f:
        f.write(','.join(map(str, bounding_box + keypoints.flatten().tolist())) + '\n')

def clamp_bbox(bbox):
    """
    Clamp bounding box values to ensure they remain within the image boundaries.

    Args:
        bbox (list): Bounding box [x1, y1, x2, y2].

    Returns:
        list: Clamped bounding box.
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, IMG_WIDTH))
    y1 = max(0, min(y1, IMG_HEIGHT))
    x2 = max(0, min(x2, IMG_WIDTH))
    y2 = max(0, min(y2, IMG_HEIGHT))
    return [x1, y1, x2, y2]

def augment_bounding_box(bounding_box):
    """
    Apply random offset to the bounding box and ensure it stays in valid bounds.

    Args:
        bounding_box (list): Original bounding box [x1, y1, x2, y2].

    Returns:
        list: Augmented bounding box [x1, y1, x2, y2].
    """
    x_offset = random.randint(*OFFSET_RANGE)
    y_offset = random.randint(*OFFSET_RANGE)

    # Apply offsets
    augmented_box = [
        bounding_box[0] + x_offset,  # x1
        bounding_box[1] + y_offset,  # y1
        bounding_box[2] + x_offset,  # x2
        bounding_box[3] + y_offset   # y2
    ]

    # Ensure bounding box stays within valid image range
    return clamp_bbox(augmented_box)

def augment_and_save_keypoints():
    """
    Augment bounding boxes and save them with keypoints.
    """
    keypoints_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith('.txt')])

    num_augmentations = int(AUGMENTATION_MULTIPLIER)  # Full augmentations (e.g., 2x)
    extra_augmentations = int(len(keypoints_files) * (AUGMENTATION_MULTIPLIER - num_augmentations))  # 0.75x

    for keypoint_file in keypoints_files:
        file_path = os.path.join(INPUT_FOLDER, keypoint_file)
        bounding_box, keypoints = load_keypoints(file_path)

        # Apply full augmentations
        for i in range(num_augmentations):
            augmented_bbox = augment_bounding_box(bounding_box)
            save_augmented_keypoints(augmented_bbox, keypoints, OUTPUT_FOLDER, f"{keypoint_file[:-4]}_aug{i}.txt")

    # Apply partial augmentation (e.g., 0.75x) on a random subset
    subset_files = random.sample(keypoints_files, extra_augmentations)
    for keypoint_file in subset_files:
        file_path = os.path.join(INPUT_FOLDER, keypoint_file)
        bounding_box, keypoints = load_keypoints(file_path)
        augmented_bbox = augment_bounding_box(bounding_box)
        save_augmented_keypoints(augmented_bbox, keypoints, OUTPUT_FOLDER, f"{keypoint_file[:-4]}_aug_extra.txt")

    print(f"Augmentation completed. Augmented keypoints saved in {OUTPUT_FOLDER}")

# Run augmentation
augment_and_save_keypoints()
