import cv2

def trim_video(input_file, output_file, start_time, end_time):
    """
    Trims a video using OpenCV.

    Args:
        input_file (str): The path to the input video file.
        output_file (str): The path to save the trimmed video.
        start_time (int): The start time in seconds.
        end_time (int): The end time in seconds.
    """

    cap = cv2.VideoCapture(input_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # You can change the codec here
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count >= start_frame and frame_count <= end_frame:
            out.write(frame)
        elif frame_count > end_frame:
            break

    cap.release()
    out.release()
    print("Video trimmed successfully!")

if __name__ == "__main__":
    input_video = "test_data/vid6.avi"
    output_video = "test_data/vid8.avi"
    start_time = 6  # Start trimming at 10 seconds
    end_time = 8   # End trimming at 30 seconds

    trim_video(input_video, output_video, start_time, end_time)