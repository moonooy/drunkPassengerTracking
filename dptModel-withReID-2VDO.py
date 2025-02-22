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
        self.pose_model = YOLO(yolo_model_path)
        self.lstm_model = load_model(lstm_model_path)
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.person_buffers = defaultdict(lambda: deque(maxlen=sequence_length))
        self.person_states = {}  # Store drunk/normal states
        self.executor = ThreadPoolExecutor(max_workers=8)  # Use threading for concurrency

    def extract_keypoints(self, frame):
        """
        Extract keypoints for detected people in the frame.
        """
        results = self.pose_model.predict(frame, save=False, device=0)  # Use GPU
        keypoints_dict = {}
        boxes = results[0].boxes
        identities = results[0].id if hasattr(results[0], 'id') else None

        if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
            keypoints_tensor = results[0].keypoints.data.cpu().numpy()
            for i, keypoints in enumerate(keypoints_tensor):
                if keypoints.shape[0] == 0:
                    keypoints = np.zeros((17, 3))
                elif keypoints.shape[0] != 17:
                    padded_keypoints = np.zeros((17, 3))
                    padded_keypoints[:keypoints.shape[0], :] = keypoints
                    keypoints = padded_keypoints

                person_id = identities[i] if identities is not None else i
                keypoints_dict[person_id] = keypoints.flatten()
                logger.debug(f"Person {person_id} Keypoints: {keypoints.shape}")
        else:
            logger.warning("⚠️ No keypoints detected in the frame.")

        return keypoints_dict, boxes

    def classify_behavior_batch(self, keypoints_dict):
        """
        Classify behavior for multiple people in batch.
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
        Process a single video frame and annotate it.
        """
        keypoints_dict, boxes = self.extract_keypoints(frame)

        classifications = self.classify_behavior_batch(keypoints_dict)
        self.person_states.update(classifications)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist()[:4])
            person_id = box.id if box.id is not None else i
            state_color = (0, 255, 0)  # Default green for "Normal"

            if person_id in self.person_states:
                is_drunk = self.person_states[person_id]
                state_color = (0, 0, 255) if is_drunk else (0, 255, 0)
                label = f"Person {person_id}: {'Drunk' if is_drunk else 'Normal'}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), state_color, 2)

        for person_id, keypoints in keypoints_dict.items():
            kp_array = keypoints.reshape(-1, 3)
            is_drunk = self.person_states.get(person_id, False)
            color = (0, 0, 255) if is_drunk else (0, 255, 0)

            for x, y, conf in kp_array:
                if conf > 0.1:
                    cv2.circle(frame, (int(x), int(y)), 5, color, -1)

        return frame


def process_video(video_path, window_name, drunk_detector):
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

    drunk_detector = DrunkDetectionReID(
        yolo_model_path=config['yolo_model_path'],
        lstm_model_path=config['lstm_model_path'],
        sequence_length=config['sequence_length'],
        confidence_threshold=config['confidence_threshold']
    )

    video_paths = [
        config['video_path1'],  # First test video
        config['video_path2']   # Second test video
    ]

    window_names = ["Drunk Detection - Video 1", "Drunk Detection - Video 2"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(process_video, video_paths[i], window_names[i], drunk_detector)
            for i in range(2)
        ]
        for future in futures:
            future.result()  # Wait for both videos to finish processing

if __name__ == "__main__":
    main('drunk-detection.yaml')
