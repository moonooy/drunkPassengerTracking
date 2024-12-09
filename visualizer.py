import cv2
import numpy as np

def overlay_keypoints_on_frame(frame_path, keypoints_path, output_path):
    # Load frame
    frame = cv2.imread(frame_path)

    # Load keypoints
    with open(keypoints_path, 'r') as f:
        keypoints = np.array([list(map(float, line.strip().split(','))) for line in f])

    # Draw keypoints
    for kp in keypoints:
        x, y, conf = kp[:3]
        if conf > 0.5:  # Draw only if confidence is high
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Save or display frame with keypoints
    cv2.imwrite(output_path, frame)
    cv2.imshow('Keypoints Overlay', frame)
    cv2.waitKey(0)

# Example usage
overlay_keypoints_on_frame(
    'extracted_frames/frames/frame_0.jpg',
    'extracted_frames/keypoints/keypoints_0.txt',
    'output/frame_0_with_keypoints.jpg'
)
