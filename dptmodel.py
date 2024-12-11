import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import deque
import yaml


class DrunkDetection:
    def __init__(self, yolo_model_path, lstm_model_path, sequence_length=30, confidence_threshold=0.5):
        """
        Initialize the Drunk Detection pipeline.

        Args:
            yolo_model_path (str): Path to the YOLO-pose model.
            lstm_model_path (str): Path to the trained LSTM model.
            sequence_length (int): Number of frames for the LSTM sequence.
            confidence_threshold (float): Threshold for classifying as drunk behavior.
        """
        self.pose_model = YOLO(yolo_model_path)
        self.lstm_model = load_model(lstm_model_path)
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.person_buffers = {}  # Buffer for each person's keypoints

    def extract_keypoints(self, frame):
        """
        Extract keypoints for detected people in the frame.

        Args:
            frame (numpy.ndarray): Input frame.

        Returns:
            dict: Keypoints indexed by person IDs.
        """
        results = self.pose_model.predict(frame, save=False)
        keypoints_dict = {}

        print("Debug: YOLO results:", results[0])  # Check YOLO results object

        if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
            # Keypoints exist
            keypoints_tensor = results[0].keypoints.data.cpu().numpy()  # Extract keypoints data explicitly
            print("Debug: Extracted keypoints array:", keypoints_tensor)

            for i, keypoints in enumerate(keypoints_tensor):
                # Skip if no keypoints detected (shape is 0)
                if keypoints.size == 0:
                    print(f"Warning: No keypoints detected for person {i}. Skipping.")
                    continue

                # Flatten dimensions if necessary
                keypoints = np.squeeze(keypoints)  # Remove any extra dimensions (e.g., (1, 17, 3) -> (17, 3))

                # Pad keypoints to ensure 17 keypoints
                if keypoints.shape[0] != 17:
                    print(f"Warning: Keypoints count mismatch ({keypoints.shape[0]}). Padding to 17.")
                    padded_keypoints = np.zeros((17, 3))
                    padded_keypoints[:keypoints.shape[0], :] = keypoints
                    keypoints = padded_keypoints

                person_id = i  # Assign an incremental ID
                keypoints_dict[person_id] = keypoints.flatten()
        else:
            print("Warning: No keypoints detected in the frame.")

        return keypoints_dict

    def classify_behavior(self, person_id, keypoints):
        """
        Classify behavior for a given person based on their sequence of keypoints.

        Args:
            person_id (int): Person ID.
            keypoints (numpy.ndarray): Flattened keypoints.

        Returns:
            bool: True if drunk behavior is detected, False otherwise.
        """
        if person_id not in self.person_buffers:
            self.person_buffers[person_id] = deque(maxlen=self.sequence_length)

        self.person_buffers[person_id].append(keypoints)
        print(f"Debug: Current buffer for person {person_id}: {len(self.person_buffers[person_id])} frames")

        # Only classify if we have enough frames
        if len(self.person_buffers[person_id]) == self.sequence_length:
            sequence = np.array(self.person_buffers[person_id]).reshape(1, self.sequence_length, -1)
            print(f"Debug: Sequence shape for classification: {sequence.shape}")

            prediction = self.lstm_model.predict(sequence)[0][0]  # Sigmoid output
            print(f"Debug: LSTM prediction for person {person_id}: {prediction}")
            return prediction > self.confidence_threshold  # Drunk if confidence > threshold

        return False

    def process_frame(self, frame):
        """
        Process a single video frame and annotate it with detection results.

        Args:
            frame (numpy.ndarray): Input video frame.

        Returns:
            numpy.ndarray: Annotated frame.
        """
        keypoints_dict = self.extract_keypoints(frame)
        for person_id, keypoints in keypoints_dict.items():
            is_drunk = self.classify_behavior(person_id, keypoints)
            label = f"Person {person_id}: {'Drunk' if is_drunk else 'Normal'}"
            color = (0, 0, 255) if is_drunk else (0, 255, 0)

            # Annotate frame with keypoints and classification
            cv2.putText(frame, label, (10, 30 * (person_id + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Visualize keypoints for debugging
            kp_array = keypoints.reshape(-1, 3)  # Reshape to (17, 3) format
            for x, y, conf in kp_array:
                if conf > 0.1:  # Only show keypoints with high confidence
                    cv2.circle(frame, (int(x), int(y)), 5, color, -1)

        return frame


def main(config_path):
    # Load configuration from YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Initialize DrunkDetection pipeline
    drunk_detector = DrunkDetection(
        yolo_model_path=config['yolo_model_path'],
        lstm_model_path=config['lstm_model_path'],
        sequence_length=config['sequence_length'],
        confidence_threshold=config['confidence_threshold']
    )

    # Open video capture
    cap = cv2.VideoCapture(config['video_path'])
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for consistent processing
        frame = cv2.resize(frame, (640, 480))

        # Process the frame
        annotated_frame = drunk_detector.process_frame(frame)

        # Display the frame
        cv2.imshow('Drunk Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main('drunk-detection.yaml')
