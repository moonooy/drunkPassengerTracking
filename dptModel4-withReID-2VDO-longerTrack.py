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
        
        self.tracker = DeepSort(
            max_age=300,  # 5 minutes
            n_init=3,  # Require 3 detections before tracking starts
            max_cosine_distance=0.2,  # Lower means more strict tracking
            nn_budget=100,  # Controls feature memory (higher = better tracking)
            # embedder="torchreid"  # (Keep None for now, but test mobilenet/torchreid)
        )

        self.person_buffers = defaultdict(lambda: deque(maxlen=sequence_length))
        self.person_states = {}

    def extract_keypoints_and_track(self, frame):
        """
        Extracts keypoints and tracks using DeepSORT.
        """
        results = self.pose_model.track(frame, persist=True, device=0, classes=0)  # ✅ Use built-in YOLO tracking
        detections = []
        keypoints_dict = {}

        for r in results:
            if hasattr(r, 'boxes') and r.boxes is not None:
                for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
                    conf = r.boxes.conf[i].item()
                    if conf > self.confidence_threshold:
                        detections.append((box, conf))  # (bbox, confidence)

        # ✅ Update DeepSORT tracker
        tracked_objects = self.tracker.update_tracks(detections, frame=frame)

        # ✅ Extract keypoints associated with track IDs
        if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
            keypoints_tensor = results[0].keypoints.data.cpu().numpy()
            for i, track in enumerate(tracked_objects):
                if track.is_confirmed():
                    track_id = track.track_id
                    keypoints = keypoints_tensor[i] if i < len(keypoints_tensor) else np.zeros((17, 3))
                    keypoints_dict[track_id] = keypoints.flatten()

        return keypoints_dict, tracked_objects

    def classify_behavior_batch(self, keypoints_dict):
        """
        Classify behavior for tracked individuals.
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
        Processes a single frame, tracks, and annotates.
        """
        keypoints_dict, tracked_objects = self.extract_keypoints_and_track(frame)
        classifications = self.classify_behavior_batch(keypoints_dict)
        self.person_states.update(classifications)

        for track in tracked_objects:
            if track.is_confirmed():
                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltwh())  # Get bbox
                
                state_color = (0, 255, 0)  # Default green for "Normal"
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

    # ✅ Separate instances for each video
    drunk_detector1 = DrunkDetectionReID(
        yolo_model_path=config['yolo_model_path'],
        lstm_model_path=config['lstm_model_path']
    )

    drunk_detector2 = DrunkDetectionReID(
        yolo_model_path=config['yolo_model_path'],
        lstm_model_path=config['lstm_model_path']
    )

    video_paths = [config['video_path1'], config['video_path2']]
    window_names = ["Drunk Detection - Video 1", "Drunk Detection - Video 2"]

    # ✅ Ensure each video gets its own tracker
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(process_video, video_paths[i], window_names[i], detector)
            for i, detector in enumerate([drunk_detector1, drunk_detector2])
        ]
        for future in futures:
            future.result()

if __name__ == "__main__":
    main('drunk-detection.yaml')
