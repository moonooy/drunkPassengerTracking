import os
import numpy as np

def filter_keypoints(keypoints, confidence_threshold=0.1):
    """
    Filters out invalid keypoints based on confidence and (0, 0) coordinates.

    Args:
        keypoints (np.array): Array of keypoints (x, y, confidence).
        confidence_threshold (float): Minimum confidence to retain a keypoint.

    Returns:
        np.array: Filtered keypoints with invalid entries removed.
    """
    filtered_keypoints = []
    for kp in keypoints:
        x, y, conf = kp
        if conf > confidence_threshold and not (x == 0 and y == 0):
            filtered_keypoints.append(kp)
    return np.array(filtered_keypoints)


def clean_keypoints_folder(keypoints_folder, output_folder, confidence_threshold=0.1):
    """
    Cleans keypoints files by removing invalid keypoints.

    Args:
        keypoints_folder (str): Path to the folder containing keypoint .txt files.
        output_folder (str): Path to save the cleaned keypoints.
        confidence_threshold (float): Minimum confidence to retain a keypoint.
    """
    os.makedirs(output_folder, exist_ok=True)
    keypoint_files = sorted([os.path.join(keypoints_folder, f) for f in os.listdir(keypoints_folder) if f.endswith('.txt')])

    for keypoint_file in keypoint_files:
        # Read keypoints from file
        with open(keypoint_file, 'r') as f:
            line = f.readline().strip()
            values = list(map(float, line.split(',')))
            keypoints = np.array([values[i:i+3] for i in range(0, len(values), 3)])  # Group into (x, y, confidence)

        # Filter keypoints
        filtered_keypoints = filter_keypoints(keypoints, confidence_threshold)

        # Save cleaned keypoints
        output_file = os.path.join(output_folder, os.path.basename(keypoint_file))
        with open(output_file, 'w') as f:
            for kp in filtered_keypoints:
                f.write(','.join(map(str, kp)) + '\n')

    print(f"Keypoints cleaned and saved to {output_folder}")


# Example usage
clean_keypoints_folder(
    keypoints_folder='extracted_frames_normal_behavior/keypoints',  # Original keypoints folder
    output_folder='cleaned_keypoints_normal_behaviors/',              # Output folder for cleaned files
    confidence_threshold=0.1                        # Confidence threshold
)
