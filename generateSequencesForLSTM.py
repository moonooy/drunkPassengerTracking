import os
import numpy as np


def load_keypoints(keypoints_folder, sequence_length=30):
    """
    Load keypoints from .txt files and organize them into sequences.
    
    Args:
        keypoints_folder (str): Path to the folder containing keypoint .txt files.
        sequence_length (int): Number of consecutive frames per sequence.
    
    Returns:
        list: A list of sequences where each sequence is an array of shape (sequence_length, num_keypoints).
    """
    # List all keypoint files in sorted order (to maintain frame order)
    keypoint_files = sorted([os.path.join(keypoints_folder, f) for f in os.listdir(keypoints_folder) if f.endswith('.txt')])
    
    # Load keypoints
    keypoints = []
    for file in keypoint_files:
        with open(file, 'r') as f:
            frame_keypoints = [list(map(float, line.strip().split(','))) for line in f]
            keypoints.append(frame_keypoints[0])  # Assuming one person per frame

    # Convert to NumPy array for easier processing
    keypoints = np.array(keypoints)  # Shape: (num_frames, num_keypoints)
    
    # Create sequences
    sequences = []
    for i in range(len(keypoints) - sequence_length + 1):
        sequences.append(keypoints[i:i + sequence_length])  # Shape: (sequence_length, num_keypoints)
    
    return sequences

# Example Usage
keypoints_folder = 'extracted_frames/keypoints'  # Path where keypoints .txt files are saved
sequences = load_keypoints(keypoints_folder, sequence_length=30)  # 30 frames per sequence
print(f"Generated {len(sequences)} sequences of shape {sequences[0].shape}.")
