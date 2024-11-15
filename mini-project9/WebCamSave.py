"""
Video Capture and Save Utility
Usage: python video_processor.py -f video_file_name -o out_video.avi
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class VideoConfig:
    """Configuration class for video processing parameters"""
    width: int
    height: int
    fps: float = 20.0
    codec: str = 'XVID'
    output_color: bool = True

class VideoProcessor:
    """Handles video capture, processing, and saving operations"""
    
    def __init__(self, input_source: Optional[str] = None, output_file: Optional[str] = None):
        """Initialize video processor with input and output sources"""
        self.input_source = 0 if input_source is None else input_source
        self.output_file = output_file
        self.capture = None
        self.writer = None
        self.config = None

    def setup_capture(self) -> bool:
        """Initialize video capture and configure parameters"""
        try:
            self.capture = cv2.VideoCapture(self.input_source)
            if not self.capture.isOpened():
                raise ValueError(f"Failed to open video source: {self.input_source}")

            # Allow camera sensor to warm up
            if isinstance(self.input_source, int):
                time.sleep(2.0)

            # Configure video parameters
            self.config = VideoConfig(
                width=int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
            return True

        except Exception as e:
            print(f"Error setting up video capture: {str(e)}")
            return False

    def setup_writer(self) -> bool:
        """Initialize video writer if output file is specified"""
        if self.output_file:
            try:
                fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
                self.writer = cv2.VideoWriter(
                    self.output_file,
                    fourcc,
                    self.config.fps,
                    (self.config.width, self.config.height),
                    self.config.output_color
                )
                return True
            except Exception as e:
                print(f"Error setting up video writer: {str(e)}")
                return False
        return True

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process the video frame. Override this method for custom processing."""
        # Add your custom frame processing here
        return frame

    def run(self):
        """Main processing loop"""
        try:
            if not self.setup_capture() or not self.setup_writer():
                return

            while True:
                ret, frame = self.capture.read()
                if not ret:
                    break

                # Process frame
                processed_frame = self.process_frame(frame)

                # Write frame if output is specified
                if self.writer:
                    self.writer.write(processed_frame)

                # Display frame
                cv2.imshow("Frame", processed_frame)

                # Check for 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"Error during video processing: {str(e)}")

        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources and cleanup"""
        if self.capture:
            self.capture.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Video processing utility for capture and save operations"
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Path to the input video file"
    )
    parser.add_argument(
        "-o", "--out",
        type=str,
        help="Output video file name"
    )
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Validate output path if specified
    if args.out:
        output_dir = Path(args.out).parent
        output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize and run video processor
    processor = VideoProcessor(args.file, args.out)
    processor.run()

if __name__ == "__main__":
    main()