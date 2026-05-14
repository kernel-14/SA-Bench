import os
import cv2

def preprocess_video(video_path, output_path):
    """Preprocess video data by resizing and normalizing."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (256, 256))
        normalized_frame = frame / 255.0
        output_file = os.path.join(output_path, f"frame_{frame_count:04d}.npy")
        np.save(output_file, normalized_frame)
        frame_count += 1
    cap.release()

if __name__ == "__main__":
    video_path = "data/example.mp4"
    output_path = "data/processed/"
    os.makedirs(output_path, exist_ok=True)
    preprocess_video(video_path, output_path)

