import os
import cv2
import argparse
from pathlib import Path
import logging

def extract_frames(video_path, output_dir, interval_sec=0.5):
    """
    Extracts frames from a video at specified intervals as close as possible
    to the specified time interval.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where extracted frames will be saved.
        interval_sec (float): Time interval between frames in seconds.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Cannot open video file {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        logging.error("Error: FPS value is 0. Cannot proceed with frame extraction.")
        cap.release()
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    logging.info(f"Video FPS: {fps}")
    logging.info(f"Total Frames: {total_frames}")
    logging.info(f"Video Duration (s): {duration_sec}")

    # Calculate frame interval
    frame_interval = int(fps * interval_sec)
    if frame_interval == 0:
        frame_interval = 1  # Ensure at least every frame is captured if interval_sec is very small

    logging.info(f"Extracting every {frame_interval} frames (every {interval_sec} seconds)")

    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if frame_count % frame_interval == 0:
            # Save the frame as an image file
            frame_filename = os.path.join(output_dir, f"frame_{extracted_count:04d}.jpg")
            success = cv2.imwrite(frame_filename, frame)
            if success:
                logging.info(f"Saved {frame_filename}")
                extracted_count += 1
            else:
                logging.warning(f"Failed to save {frame_filename}")

        frame_count += 1

    cap.release()
    logging.info(f"Extraction complete. {extracted_count} frames saved to {output_dir}.")

def main():
    """
    Main function to parse command-line arguments and initiate frame extraction.
    """
    parser = argparse.ArgumentParser(
        description='Extract frames from a video at specified intervals.'
    )    
    # Positional argument: video_path
    parser.add_argument('video_path',type=Path,
        help='Path to the input video file.'
    )
    # Positional argument: output_dir
    parser.add_argument('output_dir',type=Path,
        help='Directory to save the extracted frames.'
    )
    # Optional argument: interval_sec
    parser.add_argument('--interval_sec',type=float,default=0.5,
        help='Time interval between frames in seconds (default: 0.5).'
    )
    # Optional argument: verbose
    parser.add_argument('-v', '--verbose',action='store_true',
        help='Enable verbose logging output.'
    )
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Convert Path objects to strings for compatibility
    video_path_str = str(args.video_path)
    output_dir_str = str(args.output_dir)
    interval_sec = args.interval_sec
    
    # Validate video path
    if not Path(video_path_str).is_file():
        logging.error(f"Error: The video file '{video_path_str}' does not exist.")
        return
    
    # Validate output directory (it will be created if it doesn't exist)
    if not os.access(os.path.dirname(output_dir_str) or '.', os.W_OK):
        logging.error(f"Error: Cannot write to the directory '{output_dir_str}'. Check permissions.")
        return
    
    # Call the extract_frames function
    extract_frames(video_path_str, output_dir_str, interval_sec)

if __name__ == "__main__":
    main()
