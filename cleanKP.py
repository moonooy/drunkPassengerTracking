import os
import numpy as np

def filter_and_pad_keypoints(data, confidence_threshold=0.1):
    """
    Filters and pads keypoints while keeping bounding box coordinates.

    Args:
        data (list): List containing bounding box (first 4 values) and keypoints (remaining values).
        confidence_threshold (float): Minimum confidence to retain a keypoint.

    Returns:
        np.array: Cleaned keypoints with bounding box and shape correction.
    """
    # ✅ Extract bounding box (first 4 values)
    bbox = np.array(data[:4], dtype=float)  # Bounding box (x1, y1, x2, y2)

    # ✅ Extract keypoints (remaining values)
    keypoints = np.array(data[4:], dtype=float).reshape(-1, 3)  # (x, y, confidence)

    # ✅ Filter out low-confidence keypoints and invalid (0,0) ones
    valid_keypoints = [
        kp for kp in keypoints if kp[2] > confidence_threshold and not (kp[0] == 0 and kp[1] == 0)
    ]
    valid_keypoints = np.array(valid_keypoints)

    # ✅ Ensure 17 keypoints exist (zero-pad if missing)
    cleaned_keypoints = np.zeros((17, 3))  # Safe padding
    if valid_keypoints.shape[0] > 0:
        num_valid = min(valid_keypoints.shape[0], 17)
        cleaned_keypoints[:num_valid, :] = valid_keypoints[:num_valid, :]

    # ✅ Return concatenated bounding box + keypoints
    return np.concatenate((bbox, cleaned_keypoints.flatten()))

def clean_all_keypoints(root_folder="dataset_output", confidence_threshold=0.1):
    """
    Cleans all keypoints while keeping bounding box information.

    Args:
        root_folder (str): Path to the root dataset folder.
        confidence_threshold (float): Minimum confidence to retain a keypoint.
    """
    # Get all folders inside dataset_output
    subfolders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]

    for subfolder in subfolders:
        keypoints_folder = os.path.join(root_folder, subfolder, "keypoints")
        output_folder = os.path.join(root_folder, subfolder, "cleaned_keypoints")

        if os.path.exists(keypoints_folder):
            os.makedirs(output_folder, exist_ok=True)
            keypoint_files = sorted([os.path.join(keypoints_folder, f) for f in os.listdir(keypoints_folder) if f.endswith('.txt')])

            for keypoint_file in keypoint_files:
                # Read keypoints from file
                with open(keypoint_file, 'r') as f:
                    line = f.readline().strip()
                    if not line:
                        print(f"Warning: {keypoint_file} is empty. Skipping.")
                        continue

                    values = line.split(',')

                    try:
                        values = list(map(float, values))  # Convert to float
                    except ValueError:
                        print(f"Error: {keypoint_file} contains invalid values. Skipping.")
                        continue

                    # ✅ Ensure correct number of values (bounding box + 17 keypoints)
                    expected_values = 4 + (17 * 3)
                    if len(values) != expected_values:
                        print(f"Skipping file with incorrect format (expected {expected_values} values, got {len(values)}): {keypoint_file}")
                        continue

                # ✅ Clean and save keypoints
                cleaned_data = filter_and_pad_keypoints(values, confidence_threshold)
                output_file = os.path.join(output_folder, os.path.basename(keypoint_file))
                with open(output_file, 'w') as f:
                    f.write(','.join(map(str, cleaned_data)) + '\n')

            print(f"Cleaned keypoints saved to {output_folder}")

        else:
            print(f"No keypoints folder found in {subfolder}, skipping.")

# Run the script
if __name__ == "__main__":
    clean_all_keypoints()
