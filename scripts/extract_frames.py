import cv2
import os

def video_to_frames(video_path):
    if not os.path.exists(video_path):
        print(f"file {video_path} not exist")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0] 
    output_folder = os.path.join(os.path.dirname(video_path), video_name)
    os.makedirs(output_folder, exist_ok=True)
    print(f"save_to_place: {output_folder}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"cannot open source file {video_path}")
        return


    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:  
            break
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Total {frame_count} frames")


if __name__ == "__main__":
    video_path = "test_data/test7.avi" 
    video_to_frames(video_path)