import os
import numpy as np
from sklearn.model_selection import train_test_split

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
                print(f"Loaded keypoints from {keypoints_files[i + j]} with shape {keypoints.shape}")
                
                # Ensure consistent number of keypoints (e.g., 17)
                if keypoints.shape[0] != 17:
                    print(f"File {keypoints_files[i + j]} has {keypoints.shape[0]} keypoints. Padding with zeros.")
                    padded_keypoints = np.zeros((17, 3))  # Default to zeros for missing keypoints
                    padded_keypoints[:keypoints.shape[0], :] = keypoints
                    keypoints = padded_keypoints
                    print(f"Padded keypoints shape: {keypoints.shape}")
                
                sequence.append(keypoints.flatten())  # Flatten (x, y, confidence)
                print(f"Appended keypoints with flattened shape: {keypoints.flatten().shape}")
        
        print(f"Sequence shape: {np.array(sequence).shape}")
        sequences.append(np.array(sequence))
        labels.append(label)

    return sequences, labels

# Load drunk and normal behavior datasets
drunk_sequences, drunk_labels = load_keypoints('cleaned_keypoints_drunk_behaviors', label=1)
normal_sequences, normal_labels = load_keypoints('cleaned_keypoints_normal_behaviors', label=0)

# Combine datasets
sequences = drunk_sequences + normal_sequences
labels = drunk_labels + normal_labels

# Convert to numpy arrays
sequences = np.array(sequences)
labels = np.array(labels)

# Train-test split
X_train, X_temp, y_train, y_temp = train_test_split(sequences, labels, test_size=0.3, random_state=42, stratify=labels)

# Validation-test split
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")
