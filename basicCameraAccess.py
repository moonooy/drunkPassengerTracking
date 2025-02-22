import cv2
import threading

# RTSP URLs for the cameras
camera_urls = [
    "rtsp://FPTEST:EZ2REMEMBER@192.168.43.150/stream2",
    "rtsp://FPTEST:EZ2REMEMBER@192.168.43.57/stream2"
]

# Function to display a single camera stream
def display_camera(camera_id, url):
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print(f"Error: Unable to open camera {camera_id} at {url}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Unable to fetch frame from camera {camera_id}")
            break

        # Show the camera feed
        cv2.imshow(f"Camera {camera_id}", frame)

        # Close the window when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(f"Camera {camera_id}")

# Threads for each camera
threads = []
for i, url in enumerate(camera_urls):
    thread = threading.Thread(target=display_camera, args=(i, url))
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

cv2.destroyAllWindows()
