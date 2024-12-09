import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define the model
model = Sequential([
    LSTM(64, input_shape=(30, 51), return_sequences=True),  # 30 timesteps, 51 keypoints (17 joints * 3)
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(np.array(X_train), np.array(y_train), validation_split=0.2, epochs=20, batch_size=32)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(np.array(X_test), np.array(y_test))
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
