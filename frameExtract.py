import cv2
from ultralytics import YOLO
import os
import torch

# Check if GPU is available
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLO-pose model onto GPU (if available)
pose_model = YOLO('yolov8n-pose.pt')

def process_video(video_path, output_folder, frame_rate=30, mode='drunk'):
    """
    Process a video to extract frames, keypoints, and bounding boxes using YOLO pose estimation.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Root directory to save frames and keypoints.
        frame_rate (int): Number of frames to extract per second.
        mode (str): Behavior type ('drunk' or 'normal').
    """
    assert mode in ['drunk', 'normal'], "Mode must be either 'drunk' or 'normal'"

    # Define specific folders for storing data
    behavior_folder = os.path.join(output_folder, f"extracted_frames_{mode}_behavior")
    frames_folder = os.path.join(behavior_folder, "frames")
    keypoints_folder = os.path.join(behavior_folder, "keypoints")

    os.makedirs(frames_folder, exist_ok=True)
    os.makedirs(keypoints_folder, exist_ok=True)

    # Open video and get FPS
    cap = cv2.VideoCapture(video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, video_fps // frame_rate)  # Dynamically adjust interval

    print(f"Processing {mode} video: {video_path}")
    print(f"Video FPS: {video_fps}, Extracting every {frame_interval} frames.")

    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break  # Exit when video ends

        # Extract frames at the specified interval
        if frame_count % frame_interval == 0:
            results = pose_model.predict(frame, save=False, device=device)  # Run on GPU if available

            # Ensure keypoints & bounding boxes are detected
            if results[0].keypoints is not None and results[0].keypoints.data is not None:
                keypoints = results[0].keypoints.data.detach().cpu().numpy()

                # Handle bounding boxes safely
                bounding_boxes = results[0].boxes.xyxy.detach().cpu().numpy() if results[0].boxes is not None else []
                if len(bounding_boxes) == 0:
                    bounding_boxes = [[0, 0, 0, 0]] * len(keypoints)  # Assign default bbox per keypoint set

                # Save frame
                frame_path = os.path.join(frames_folder, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)

                # Save keypoints and bounding boxes together
                keypoints_path = os.path.join(keypoints_folder, f"keypoints_{frame_count}.txt")
                with open(keypoints_path, 'w') as f:
                    for i, kp in enumerate(keypoints):
                        bbox = bounding_boxes[i] if i < len(bounding_boxes) else [0, 0, 0, 0]  # Avoid out-of-bounds error
                        bbox_str = ",".join(map(str, bbox))  # Convert bbox to string format
                        keypoints_str = ",".join(map(str, kp.flatten()))  # Flatten keypoints

                        f.write(f"{bbox_str},{keypoints_str}\n")  # Save bbox + keypoints together

            else:
                print(f"Warning: No keypoints detected at frame {frame_count}.")

        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

        frame_count += 1

    cap.release()
    print(f"Processing completed. Frames saved to {frames_folder}, keypoints saved to {keypoints_folder}.")

# Run the function for both behaviors automatically
if __name__ == "__main__":
    process_video(r'nonCoding\Videos for dataset\Drunk\TRN_DRNK_COMBINED.mp4',
                  'dataset_output', frame_rate=60, mode='drunk')

    process_video(r'nonCoding\Videos for dataset\Normal\TRN_NORM_COMBINED.mp4',
                  'dataset_output', frame_rate=60, mode='normal')
