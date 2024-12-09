import cv2
from ultralytics import YOLO

# Load YOLO-pose model
pose_model = YOLO('yolov8n-pose.pt')  # Replace with your pose model

def process_video(video_path, output_folder, frame_rate=1):
    """
    Process a video to extract frames and generate keypoints.
    
    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Directory to save processed frames and keypoints.
        frame_rate (int): Number of frames to skip between extractions.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, frame = cap.read()

    while success:
        if frame_count % frame_rate == 0:
            # Run pose estimation on the frame
            results = pose_model.predict(frame, save=False)
            keypoints = results[0].keypoints.cpu().numpy()  # Extract keypoints

            # Save the frame with overlayed keypoints (optional)
            output_frame_path = f"{output_folder}/frame_{frame_count}.jpg"
            cv2.imwrite(output_frame_path, frame)

            # Save keypoints to a file
            output_keypoints_path = f"{output_folder}/keypoints_{frame_count}.txt"
            with open(output_keypoints_path, 'w') as f:
                for kp in keypoints:
                    f.write(','.join(map(str, kp.flatten())) + '\n')

        success, frame = cap.read()
        frame_count += 1

    cap.release()
    print(f"Processing completed. Data saved to {output_folder}.")

# Usage example
process_video('drunk_clip.mp4', 'output_folder', frame_rate=30)
