# Short max_age leads to shorter tracks, but more accurate tracking
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from concurrent.futures import ThreadPoolExecutor
from collections import deque, defaultdict
import yaml
import tensorflow as tf
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DrunkDetectionReID:
    def __init__(self, yolo_model_path, lstm_model_path, sequence_length=30, confidence_threshold=0.5):
        self.pose_model = YOLO(yolo_model_path)  # Load YOLO pose model
        self.lstm_model = load_model(lstm_model_path)  # Load LSTM model
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.person_buffers = defaultdict(lambda: deque(maxlen=sequence_length))  # Store keypoints for tracking
        self.person_states = {}  # Store drunk/normal classifications
        self.executor = ThreadPoolExecutor(max_workers=8)  # Thread pool for concurrency

    def extract_keypoints(self, frame):
        """
        Extracts keypoints and bounding boxes for detected people in the frame.
        Uses YOLO-Pose with tracking to maintain person IDs across frames.
        """
        results = self.pose_model.track(frame, persist=True, classes=0, device=0, conf=0.5)  # Track people with YOLO-Pose
        
        if results[0] is None or results[0].boxes is None:  # No detections found
            return {}, []

        boxes = results[0].boxes
        keypoints_dict = {}
        keypoints_tensor = results[0].keypoints.data.cpu().numpy() if results[0].keypoints is not None else None

        for i, box in enumerate(boxes):
            if box.id is not None:
                person_id = int(box.id.item())  # Use assigned ID from tracker
            else:
                person_id = i  # Fallback to index-based ID

            if keypoints_tensor is not None and i < len(keypoints_tensor):
                keypoints_dict[person_id] = keypoints_tensor[i].flatten()
            else:
                keypoints_dict[person_id] = np.zeros(51)  # Default if no keypoints found

        return keypoints_dict, boxes

    def classify_behavior_batch(self, keypoints_dict):
        """
        Classifies behavior for multiple people in a batch.
        Maintains a sequence buffer for each tracked person.
        """
        batch_sequences = []
        person_ids = []

        for person_id, keypoints in keypoints_dict.items():
            self.person_buffers[person_id].append(keypoints)
            if len(self.person_buffers[person_id]) == self.sequence_length:
                batch_sequences.append(np.array(self.person_buffers[person_id]))
                person_ids.append(person_id)

        if batch_sequences:
            batch_sequences = np.array(batch_sequences)
            predictions = self.lstm_model.predict(batch_sequences, batch_size=len(batch_sequences)).flatten()
            return {person_ids[i]: predictions[i] > self.confidence_threshold for i in range(len(person_ids))}

        return {}

    def process_frame(self, frame):
        """
        Processes a single video frame: detects people, extracts keypoints, and classifies behavior.
        """
        keypoints_dict, boxes = self.extract_keypoints(frame)

        if len(keypoints_dict) == 0:
            return frame  # Skip processing if no detections

        classifications = self.classify_behavior_batch(keypoints_dict)
        self.person_states.update(classifications)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist()[:4])
            person_id = int(box.id.item()) if box.id is not None else i
            state_color = (0, 255, 0)  # Default green for "Normal"

            if person_id in self.person_states:
                is_drunk = self.person_states[person_id]
                state_color = (0, 0, 255) if is_drunk else (0, 255, 0)
                label = f"Person {person_id}: {'Drunk' if is_drunk else 'Normal'}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), state_color, 2)

        return frame


def process_video(video_path, window_name, drunk_detector):
    """
    Processes video stream for drunk detection.
    Handles multiple videos asynchronously.
    """
    cap = cv2.VideoCapture(video_path)
    fps_start_time = time.time()
    frame_count = 0

    ret, frame = cap.read()
    if ret:
        frame_height, frame_width = frame.shape[:2]
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, frame_width, frame_height)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        annotated_frame = drunk_detector.process_frame(frame)

        elapsed_time = time.time() - fps_start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(window_name, annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)


def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    video_paths = [config['video_path1'], config['video_path2']]
    window_names = ["Drunk Detection - Video 1", "Drunk Detection - Video 2"]

    # 🔹 Create a **separate DrunkDetectionReID instance** for each video
    drunk_detectors = [
        DrunkDetectionReID(
            yolo_model_path=config['yolo_model_path'],
            lstm_model_path=config['lstm_model_path'],
            sequence_length=config['sequence_length'],
            confidence_threshold=config['confidence_threshold']
        )
        for _ in range(len(video_paths))
    ]

    with ThreadPoolExecutor(max_workers=len(video_paths)) as executor:
        futures = [
            executor.submit(process_video, video_paths[i], window_names[i], drunk_detectors[i])
            for i in range(len(video_paths))
        ]
        
        # 🔹 Wait for both threads to complete
        for future in futures:
            future.result()


if __name__ == "__main__":
    main('drunk-detection.yaml')
