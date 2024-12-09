import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO
from torchvision.models import resnet50, ResNet50_Weights

# Load pre-trained ResNet model with explicit weights
weights = ResNet50_Weights.DEFAULT
resnet_model = resnet50(weights=weights)
resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))  # Remove classification head
resnet_model.eval()

# Load YOLOv8 model
yolo_model = YOLO('yolov8n.pt')  # Change to yolov8n-pose.pt if using pose estimation

# Transform input for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize tracking variables
embeddings = []  # List to store feature embeddings for each track ID
track_ids = []   # Corresponding track IDs
current_id = 0   # Track ID counter


def extract_features(image):
    """Extract features from an image using ResNet."""
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = resnet_model(image_tensor).squeeze()  # Extract features
    return features.numpy()


def match_features(features_list, new_feature, threshold=0.7):
    """Match a new feature with existing features using cosine similarity."""
    if not features_list:
        return None, 0  # No matches if list is empty

    similarities = cosine_similarity([new_feature], features_list)
    max_similarity = max(similarities[0])
    if max_similarity > threshold:
        matched_index = similarities[0].tolist().index(max_similarity)
        return matched_index, max_similarity
    return None, 0


def process_frame(frame, results):
    """Process each frame: detect objects, extract features, and perform Re-ID."""
    global embeddings, track_ids, current_id

    # Iterate over detected objects
    for box in results.boxes.xyxy:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, box)
        cropped_image = frame[y1:y2, x1:x2]  # Crop the bounding box
        if cropped_image.size == 0:
            continue  # Skip if the crop is invalid

        # Convert cropped image to PIL format for ResNet input
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image = transforms.ToPILImage()(cropped_image)

        # Extract features from the cropped image
        new_feature = extract_features(cropped_image)

        # Match the new feature with existing ones
        matched_id, similarity = match_features(embeddings, new_feature)
        if matched_id is None:
            # New object detected, assign a new ID
            embeddings.append(new_feature)
            track_ids.append(current_id)
            matched_id = current_id
            current_id += 1

        # Draw bounding box and ID on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {matched_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


# Main script
def main():
    # Initialize video capture (use 0 for webcam)
    video_capture = cv2.VideoCapture('0')  # Replace with 0 for live camera

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Run YOLO detection on the frame
        results = yolo_model.predict(frame, stream=True)

        # Process the frame with Re-ID
        processed_frame = process_frame(frame, results)

        # Display the processed frame
        cv2.imshow('Re-ID Tracking', processed_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
