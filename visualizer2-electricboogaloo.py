import cv2
import os
import numpy as np
import re

SKELETON = [
    (0, 1),  # Nose -> Left Eye
    (1, 3),  # Left Eye -> Left Ear
    (0, 2),  # Nose -> Right Eye
    (2, 4),  # Right Eye -> Right Ear
    (5, 6),  # Left Shoulder -> Right Shoulder
    (5, 7),  # Left Shoulder -> Left Elbow
    (7, 9),  # Left Elbow -> Left Wrist
    (6, 8),  # Right Shoulder -> Right Elbow
    (8, 10), # Right Elbow -> Right Wrist
    (11, 12), # Left Hip -> Right Hip
    (5, 11),  # Left Shoulder -> Left Hip
    (6, 12),  # Right Shoulder -> Right Hip
    (11, 13), # Left Hip -> Left Knee
    (13, 15), # Left Knee -> Left Ankle
    (12, 14), # Right Hip -> Right Knee
    (14, 16)  # Right Knee -> Right Ankle
]

def natural_sort_key(file):
    """
    Sort filenames numerically based on digits.
    Ensures proper order for filenames like frame_1, frame_10, etc.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', file)]

def filter_keypoints(keypoints, confidence_threshold=0.1):
    """
    Filters out invalid keypoints based on confidence and (0, 0) coordinates.

    Args:
        keypoints (np.array): Array of keypoints (x, y, confidence).
        confidence_threshold (float): Minimum confidence to retain a keypoint.

    Returns:
        np.array: Filtered keypoints with invalid entries removed.
    """
    filtered_keypoints = []
    for kp in keypoints:
        x, y, conf = kp
        if conf > confidence_threshold and not (x == 0 and y == 0):
            filtered_keypoints.append(kp)
    return np.array(filtered_keypoints)

def visualize_frames_with_keypoints(frames_folder, keypoints_folder, delay=33):
    frame_files = sorted([os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if f.endswith('.jpg')], key=natural_sort_key)
    keypoint_files = sorted([os.path.join(keypoints_folder, f) for f in os.listdir(keypoints_folder) if f.endswith('.txt')], key=natural_sort_key)

    if len(frame_files) != len(keypoint_files):
        print(f"Warning: Mismatch in number of frames ({len(frame_files)}) and keypoints ({len(keypoint_files)}).")
        return

    for frame_file, keypoint_file in zip(frame_files, keypoint_files):
        frame = cv2.imread(frame_file)

        # Read and process keypoints from cleaned files
        with open(keypoint_file, 'r') as f:
            keypoints = np.array([list(map(float, line.strip().split(','))) for line in f if line.strip()])  # Line-by-line parsing

        h, w, _ = frame.shape
        print(f"Processing frame: {os.path.basename(frame_file)}")

        # Draw keypoints
        for i, kp in enumerate(keypoints):
            x, y, conf = kp[:3]
            if conf > 0.1 and 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            else:
                print(f"Skipping keypoint {i}: x={x}, y={y}, conf={conf}")

        # Draw skeleton
        for connection in SKELETON:
            start_idx, end_idx = connection
            if (
                start_idx < len(keypoints) and end_idx < len(keypoints)
                and keypoints[start_idx][2] > 0.1  # Confidence of start point
                and keypoints[end_idx][2] > 0.1    # Confidence of end point
            ):
                start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                if (0 <= start_point[0] < w and 0 <= start_point[1] < h and
                        0 <= end_point[0] < w and 0 <= end_point[1] < h):
                    cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
                else:
                    print(f"Skipping line: start={start_point}, end={end_point}")
            else:
                print(f"Skipping skeleton connection: {connection} due to low confidence or missing keypoints.")

        # Display the frame
        cv2.imshow('Keypoints Visualization', frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# Example usage
visualize_frames_with_keypoints(
    frames_folder='extracted_frames/frames',  # Replace with your frames folder
    keypoints_folder='cleaned_keypoints',  # Replace with your keypoints folder
    delay=33  # 30 FPS
)
