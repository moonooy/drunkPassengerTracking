import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
print("DeepSort is successfully imported!")
from tensorflow.keras.models import load_model
from collections import deque, defaultdict
import yaml
import tensorflow as tf
import time
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DrunkDetectionReID:
    def __init__(self, yolo_model_path, lstm_model_path, sequence_length=30, confidence_threshold=0.5):
        self.pose_model = YOLO(yolo_model_path)
        self.lstm_model = load_model(lstm_model_path)
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold

        # Each video gets its own independent tracker to prevent cross-video IDs
        self.tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)

        self.person_buffers = defaultdict(lambda: deque(maxlen=sequence_length))
        self.person_states = {}
        self.track_memory = {}  # Store re-ID info for long-term tracking

    def extract_keypoints_and_track(self, frame):
        """
        Extracts keypoints and tracks persons separately for each video.
        """
        results = self.pose_model.predict(frame, save=False, device=0)
        detections = []
        keypoints_dict = {}

        for r in results:
            if hasattr(r, 'boxes') and r.boxes is not None:
                for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
                    conf = r.boxes.conf[i].item()
                    if conf > self.confidence_threshold:
                        detections.append((box, conf))

        tracked_objects = self.tracker.update_tracks(detections, frame=frame)

        if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
            keypoints_tensor = results[0].keypoints.data.cpu().numpy()
            for i, track in enumerate(tracked_objects):
                if track.is_confirmed() and i < len(keypoints_tensor):
                    track_id = track.track_id
                    keypoints_dict[track_id] = keypoints_tensor[i].flatten()

        return keypoints_dict, tracked_objects

    def classify_behavior_batch(self, keypoints_dict):
        batch_sequences = []
        person_ids = []

        if not keypoints_dict:  # If no person is detected, return immediately
            return {}

        for person_id, keypoints in keypoints_dict.items():
            # Ensure keypoints are numpy arrays
            keypoints = np.array(keypoints, dtype=np.float32)

            self.person_buffers[person_id].append(keypoints)

            if len(self.person_buffers[person_id]) == self.sequence_length:
                sequence = np.array(list(self.person_buffers[person_id]))

                # Ensure correct shape before adding to batch
                expected_shape = (self.sequence_length, keypoints.shape[0])
                if sequence.shape == expected_shape:
                    batch_sequences.append(sequence)
                    person_ids.append(person_id)
                else:
                    logger.warning(f"Skipping inconsistent sequence for Person {person_id}: {sequence.shape}")

        if batch_sequences:
            batch_sequences = np.array(batch_sequences, dtype=np.float32)
            predictions = self.lstm_model.predict(batch_sequences, batch_size=len(batch_sequences)).flatten()
            return {person_ids[i]: predictions[i] > self.confidence_threshold for i in range(len(person_ids))}

        return {}

    def process_frame(self, frame):
        """
        Processes a single frame, tracks, and annotates detected persons.
        Handles cases where no persons are detected to prevent crashes.
        """
        keypoints_dict, tracked_objects = self.extract_keypoints_and_track(frame)

        # If no persons detected, return the original frame without crashing
        if not tracked_objects:
            logger.info("No persons detected in this frame.")
            return frame  

        # Ensure only valid sequences are passed
        classifications = self.classify_behavior_batch(keypoints_dict) if keypoints_dict else {}
        self.person_states.update(classifications)

        # Draw bounding boxes and labels
        for track in tracked_objects:
            if track.is_confirmed():
                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltwh())  # Bounding box coordinates
                state_color = (0, 255, 0)  # Default: Green (Normal)
                label = f"Person {track_id}: Normal"

                if track_id in self.person_states:
                    is_drunk = self.person_states[track_id]
                    state_color = (0, 0, 255) if is_drunk else (0, 255, 0)
                    label = f"Person {track_id}: {'Drunk' if is_drunk else 'Normal'}"

                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), state_color, 2)

        return frame

def process_video(video_path, window_name, drunk_detector):
    cap = cv2.VideoCapture(video_path)
    fps_start_time = time.time()
    frame_count = 0

    ret, frame = cap.read()
    if ret:
        frame_height, frame_width = frame.shape[:2]
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        annotated_frame = drunk_detector.process_frame(frame)

        elapsed_time = time.time() - fps_start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow(window_name, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)

def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # video_paths = [config.get('video_path1'), config.get('video_path2')]
    video_paths = [config.get('video_path1')]  
    video_paths = [v for v in video_paths if v]  # Remove any None values if a video path is missing
    window_names = [f"Drunk Detection - Video {i+1}" for i in range(len(video_paths))]

    if not video_paths:
        logger.error("No valid video paths found in the configuration file.")
        return

    # Create separate instances of DrunkDetectionReID for each video
    drunk_detectors = [
        DrunkDetectionReID(
            yolo_model_path=config['yolo_model_path'],
            lstm_model_path=config['lstm_model_path']
        ) for _ in range(len(video_paths))
    ]

    with ThreadPoolExecutor(max_workers=len(video_paths)) as executor:
        futures = [
            executor.submit(process_video, video_paths[i], window_names[i], drunk_detectors[i])
            for i in range(len(video_paths))
        ]

        for future in futures:
            future.result()  # Ensures all threads complete before the script exits

if __name__ == "__main__":
    main('drunk-detection.yaml')
