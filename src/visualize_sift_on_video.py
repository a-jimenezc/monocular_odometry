import cv2

import cv2

def visualize_sift_on_video(video_path, output_path, output_fps=None):
    """
    Visualizes SIFT features on top of video frames and saves the output.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the output video.
        output_fps (float): Frame rate of the output video. If None, use the input video's FPS.
    """
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if output_fps is None else output_fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4

    # Initialize video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Convert frame to grayscale (SIFT works on grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect SIFT features
        keypoints, _ = sift.detectAndCompute(gray, None)

        # Draw keypoints on the original frame
        frame_with_keypoints = cv2.drawKeypoints(
            frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        # Write the frame to the output video
        out.write(frame_with_keypoints)

        # Optionally display the frame with keypoints
        cv2.imshow("SIFT Features", frame_with_keypoints)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

