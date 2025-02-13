import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 🔹 Step 1: Load and Normalize Dataset
def load_keypoints(folder, label, sequence_length=30, img_width=1920, img_height=1080):
    keypoints_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.txt')])
    sequences = []
    labels = []

    for i in range(0, len(keypoints_files) - sequence_length + 1, sequence_length):
        sequence = []
        for j in range(sequence_length):
            with open(keypoints_files[i + j], 'r') as f:
                keypoints = np.array([list(map(float, line.strip().split(','))) for line in f])

                # ✅ Ensure correct shape (17, 3)
                if keypoints.shape != (17, 3):
                    padded_keypoints = np.zeros((17, 3))
                    num_valid = min(keypoints.shape[0], 17)
                    padded_keypoints[:num_valid, :] = keypoints[:num_valid, :]
                    keypoints = padded_keypoints

                # ✅ Normalize keypoints
                keypoints[:, 0] /= img_width   # Normalize x-coordinates
                keypoints[:, 1] /= img_height  # Normalize y-coordinates

                sequence.append(keypoints.flatten())

        sequences.append(np.array(sequence))
        labels.append(label)

    return sequences, labels

# 🔹 Load and Combine Datasets
drunk_sequences, drunk_labels = load_keypoints(r'dataset_output\extracted_frames_drunk_behavior\cleaned_keypoints', label=1)
normal_sequences, normal_labels = load_keypoints(r'dataset_output\extracted_frames_normal_behavior\augmented_keypoints', label=0)

sequences = np.array(drunk_sequences + normal_sequences)
labels = np.array(drunk_labels + normal_labels)

# 🔹 Shuffle and Split Data
X_train, X_temp, y_train, y_temp = train_test_split(sequences, labels, test_size=0.3, random_state=42, stratify=labels, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp, shuffle=True)

# 🔹 Print Dataset Info
print(f"Train: {sum(y_train == 0)} normal, {sum(y_train == 1)} drunk")
print(f"Validation: {sum(y_val == 0)} normal, {sum(y_val == 1)} drunk")
print(f"Test: {sum(y_test == 0)} normal, {sum(y_test == 1)} drunk")
print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}")

# 🔹 Step 2: Define Optimized LSTM Model
def create_lstm_model(input_shape):
    model = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),  # Ignore zero-padded values
        LSTM(128, return_sequences=True, activation='tanh'),
        BatchNormalization(),  # Stabilize training
        Dropout(0.3),
        LSTM(64, activation='tanh'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification (Normal vs Drunk)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

input_shape = (X_train.shape[1], X_train.shape[2])
model = create_lstm_model(input_shape)
model.summary()

# 🔹 Step 3: Train the Model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,  # Increased batch size for stability
    callbacks=[early_stopping, lr_scheduler]
)

# 🔹 Step 4: Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# 🔹 Step 5: Analyze Performance
y_pred_probs = model.predict(X_test).flatten()  # Get probabilities
y_pred = (y_pred_probs > 0.5).astype("int32")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Drunk']))

# 🔹 Step 6: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Drunk'], yticklabels=['Normal', 'Drunk'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 🔹 Step 7: ROC Curve & AUC Score
fpr, tpr, _ = roc_curve(y_test, y_pred_probs)  # Fixed ROC curve calculation
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# 🔹 Step 8: Save the Model
model.save('CUSTOM_drunk_behavior_lstm.h5')
print("\nModel saved as CUSTOM_drunk_behavior_lstm.h5")
