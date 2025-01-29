import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Step 1: Load and Verify Dataset
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
                
                # Ensure consistent number of keypoints (e.g., 17)
                if keypoints.shape[0] != 17:
                    padded_keypoints = np.zeros((17, 3))  # Default to zeros for missing keypoints
                    padded_keypoints[:keypoints.shape[0], :] = keypoints
                    keypoints = padded_keypoints
                
                sequence.append(keypoints.flatten())  # Flatten (x, y, confidence)
        
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

# Verify dataset
print(f"Train set: {sum(y_train == 0)} normal, {sum(y_train == 1)} drunk")
print(f"Validation set: {sum(y_val == 0)} normal, {sum(y_val == 1)} drunk")
print(f"Test set: {sum(y_test == 0)} normal, {sum(y_test == 1)} drunk")
print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}")

# Step 2: Define the LSTM Model
def create_lstm_model(input_shape):
    """
    Create an LSTM model for binary classification.

    Args:
        input_shape (tuple): Shape of the input data (sequence_length, num_features).

    Returns:
        Sequential: Compiled LSTM model.
    """
    model = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),  # Ignore zero-padded values
        LSTM(64, return_sequences=True, activation='tanh'),
        Dropout(0.2),
        LSTM(32, activation='tanh'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification: Normal (0) vs Drunk (1)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, num_features)
model = create_lstm_model(input_shape)
model.summary()

# Step 3: Train the Model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,  
    batch_size=16,
    callbacks=[early_stopping]
)

# Step 4: Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Step 5: Analyze Performance
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Drunk']))

# Step 6: Visualize Training Progress
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Step 7: Save the Model
model.save('drunk_behavior_lstm.h5')
print("Model saved as drunk_behavior_lstm.h5")
