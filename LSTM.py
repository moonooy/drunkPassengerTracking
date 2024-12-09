import os
import numpy as np

def load_keypoints(folder, label, sequence_length=30):
    """
    Load keypoints from a folder and organize them into sequences.

    Args:
        folder (str): Path to the folder containing keypoints files.
        label (int): Label for the data (0 = normal, 1 = drunk).
        sequence_length (int): Number of frames per sequence.

    Returns:
        list: Sequences of keypoints.
        list: Corresponding labels.
    """
    keypoints_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.txt')])
    sequences = []
    labels = []

    # Group frames into sequences
    for i in range(0, len(keypoints_files) - sequence_length + 1, sequence_length):
        sequence = []
        for j in range(sequence_length):
            with open(keypoints_files[i + j], 'r') as f:
                keypoints = np.array([list(map(float, line.strip().split(','))) for line in f])
                sequence.append(keypoints.flatten())  # Flatten (x, y, confidence)
        sequences.append(np.array(sequence))
        labels.append(label)

    return sequences, labels

# Load drunk and normal behavior datasets
drunk_sequences, drunk_labels = load_keypoints('cleaned_keypoints_drunk', label=1)
normal_sequences, normal_labels = load_keypoints('cleaned_keypoints_normal', label=0)

# Combine datasets
sequences = drunk_sequences + normal_sequences
labels = drunk_labels + normal_labels
