import os
import numpy as np

def filter_and_pad_keypoints(keypoints, confidence_threshold=0.1):
    """
    Filters out invalid keypoints based on confidence and ensures consistent (17,3) shape.

    Args:
        keypoints (np.array): Array of keypoints (x, y, confidence).
        confidence_threshold (float): Minimum confidence to retain a keypoint.

    Returns:
        np.array: Cleaned keypoints with missing values filled and shape corrected to (17,3).
    """
    # ✅ Remove keypoints with confidence below threshold or at (0,0)
    valid_keypoints = [kp for kp in keypoints if len(kp) == 3 and kp[2] > confidence_threshold and not (kp[0] == 0 and kp[1] == 0)]
    valid_keypoints = np.array(valid_keypoints)

    # ✅ Ensure the output is always (17,3)
    cleaned_keypoints = np.zeros((17, 3))  # Initialize with zeros (safe padding)
    
    if valid_keypoints.shape[0] > 0:
        num_valid = min(valid_keypoints.shape[0], 17)
        cleaned_keypoints[:num_valid, :] = valid_keypoints[:num_valid, :]
    
    return cleaned_keypoints


def clean_all_keypoints(root_folder="dataset_output", confidence_threshold=0.1):
    """
    Automatically finds and cleans all keypoints folders inside `dataset_output/`.

    Args:
        root_folder (str): Path to the root dataset folder containing keypoints directories.
        confidence_threshold (float): Minimum confidence to retain a keypoint.
    """
    # Get all folders inside dataset_output/
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
                        print(f"⚠️ Warning: {keypoint_file} is empty. Skipping.")
                        continue  # Skip empty files

                    values = line.split(',')
                    
                    try:
                        values = list(map(float, values))  # Convert to floats
                    except ValueError:
                        print(f"🚫 Error: {keypoint_file} contains invalid values. Skipping.")
                        continue

                    # ✅ Reshape to ensure (17,3) structure
                    if len(values) == 51:
                        keypoints = np.array(values).reshape(17, 3)
                    else:
                        print(f"⚠️ Warning: {keypoint_file} has incorrect format (Expected 51 values, got {len(values)}). Skipping.")
                        continue

                # ✅ Filter and pad keypoints
                cleaned_keypoints = filter_and_pad_keypoints(keypoints, confidence_threshold)

                # ✅ Save cleaned keypoints
                output_file = os.path.join(output_folder, os.path.basename(keypoint_file))
                with open(output_file, 'w') as f:
                    for kp in cleaned_keypoints:
                        f.write(','.join(map(str, kp)) + '\n')

            print(f"✅ Cleaned keypoints saved to {output_folder}")

        else:
            print(f"🚫 No keypoints folder found in {subfolder}, skipping.")


# Run the script
if __name__ == "__main__":
    clean_all_keypoints()
