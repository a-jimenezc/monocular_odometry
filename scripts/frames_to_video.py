import cv2
import os

def frames_to_video(input_folder, output_video_path, frame_interval, fps=30):
    
    if not os.path.exists(input_folder):
        print(f" {input_folder} folder not exits")
        return

    
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    if not frame_files:
        print(f" {input_folder} without files")
        return

    
    selected_frames = frame_files[::frame_interval]
    
    first_frame_path = os.path.join(input_folder, selected_frames[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, _ = first_frame.shape

    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for frame_file in selected_frames:
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

        video_writer.write(frame)

    video_writer.release()
    print(f"video save to: {output_video_path}")


if __name__ == "__main__":
    frame_interval = 5
    input_folder = "test_data/test7"  
    output_video_path = f"test_data/test7_composed_{frame_interval}.mp4"  
    fps = 30  
    frames_to_video(input_folder, output_video_path, frame_interval, fps)
