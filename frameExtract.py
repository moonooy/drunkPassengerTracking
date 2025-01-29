import cv2
from ultralytics import YOLO
import os

# Load YOLO-pose model onto the GPU
pose_model = YOLO('yolov8n-pose.pt')  # Replace with your pose estimation model

def process_video(video_path='nonCoding\cut-NormalBehavior1.mp4',
                  output_folder='extracted_frames_normal_behavior', frame_rate=30):
    """
    Process a video to extract frames and generate keypoints using GPU.
    
    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Directory to save processed frames and keypoints.
        frame_rate (int): Interval for extracting frames (e.g., 30 = 1 frame per second for a 30 FPS video).
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    frames_folder = os.path.join(output_folder, "frames")
    keypoints_folder = os.path.join(output_folder, "keypoints")
    os.makedirs(frames_folder, exist_ok=True)
    os.makedirs(keypoints_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get the video frame rate
    frame_interval = max(1, fps // frame_rate)  # Calculate the interval for frame extraction

    frame_count = 0
    success, frame = cap.read()

    while success:
        # Extract frames at the specified interval
        if frame_count % frame_interval == 0:
            # Run pose estimation on the frame using GPU
            results = pose_model.predict(frame, save=False, device=0)  # Device 0 = GPU
            keypoints = results[0].keypoints if results[0].keypoints is not None else []

            # Save the frame
            output_frame_path = os.path.join(frames_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(output_frame_path, frame)

            # Save keypoints
            output_keypoints_path = os.path.join(keypoints_folder, f"keypoints_{frame_count}.txt")
            with open(output_keypoints_path, 'w') as f:
                for kp in keypoints:
                    if kp.data is not None:
                        # Process keypoints on GPU, detach, and move to CPU for saving
                        keypoint_array = kp.data.detach().cpu().numpy()
                        f.write(','.join(map(str, keypoint_array.flatten())) + '\n')

        # Read the next frame
        success, frame = cap.read()
        frame_count += 1

    cap.release()
    print(f"Processing completed. Frames saved to {frames_folder}, keypoints saved to {keypoints_folder}.")

# Run the function
if __name__ == "__main__":
    process_video()
