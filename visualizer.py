import cv2
import numpy as np

def visualize_frame_with_keypoints(frame_path, keypoints_path):
    # Load frame
    frame = cv2.imread(frame_path)

    # Load keypoints
    with open(keypoints_path, 'r') as f:
        keypoints = np.array([list(map(float, line.strip().split(','))) for line in f])

    # Get frame dimensions
    h, w, _ = frame.shape

    # Overlay keypoints
    for kp in keypoints:
        if len(kp) < 3:
            print(f"Skipping invalid keypoint: {kp}")
            continue  # Skip malformed keypoints

        x, y, conf = kp[:3]
        print(f"Keypoint: x={x}, y={y}, conf={conf}")

        if not (0 <= x < w and 0 <= y < h):
            print(f"Keypoint out of bounds: x={x}, y={y}, frame width={w}, frame height={h}")
            continue

        if conf > 0.1:  # Display all keypoints with a minimum confidence
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow('Keypoints Visualization', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
visualize_frame_with_keypoints(
    frame_path='extracted_frames/frames/frame_0.jpg',
    keypoints_path='extracted_frames/keypoints/keypoints_0.txt'
)
