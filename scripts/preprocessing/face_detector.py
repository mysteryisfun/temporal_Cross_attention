#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: face_detector.py
# Description: Face detection and alignment system for Cross-Attention CNN Research

import os
import cv2
import numpy as np
import logging
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from mtcnn import MTCNN
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt

# Set up logger
def setup_logger(log_file=None):
    """
    Set up the logger for face detection process.
    
    Args:
        log_file (str, optional): Path to log file. If None, logs only to console.
        
    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger("FaceDetector")
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is provided)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class FaceProcessor:
    """
    A class for detecting, aligning, and processing faces in images.
    """
    
    def __init__(self, confidence_threshold=0.9, min_face_size=80, target_size=(224, 224)):
        """
        Initialize the FaceProcessor with detection parameters.
        
        Args:
            confidence_threshold (float): Minimum confidence for valid face detections.
            min_face_size (int): Minimum width/height for valid faces.
            target_size (tuple): Size to resize faces to (width, height).
        """
        self.detector = MTCNN()
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size
        self.target_size = target_size
    
    def detect_faces(self, image):
        """
        Detect faces in an image.
        
        Args:
            image (numpy.ndarray): Image to detect faces in.
            
        Returns:
            list: List of detected face dictionaries from MTCNN.
        """
        # Ensure image is RGB (MTCNN expects RGB)
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3 and image.dtype == np.uint8:
            # Check if image is BGR (OpenCV default)
            if cv2.COLOR_BGR2RGB != 0:  # This isn't a no-op conversion
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        try:
            faces = self.detector.detect_faces(image)
            # Filter faces by confidence and size
            valid_faces = []
            for face in faces:
                if face['confidence'] >= self.confidence_threshold:
                    x, y, w, h = face['box']
                    if w >= self.min_face_size and h >= self.min_face_size:
                        valid_faces.append(face)
            return valid_faces
        except Exception as e:
            logging.error(f"Face detection error: {str(e)}")
            return []
    
    def get_eye_centers(self, face):
        """
        Calculate eye centers from facial landmarks.
        
        Args:
            face (dict): Face dictionary from MTCNN with keypoints.
            
        Returns:
            tuple: Left eye center (x, y) and right eye center (x, y).
        """
        left_eye = face['keypoints']['left_eye']
        right_eye = face['keypoints']['right_eye']
        return left_eye, right_eye
    
    def align_face(self, image, face):
        """
        Align a face based on eye positions.
        
        Args:
            image (numpy.ndarray): Original image.
            face (dict): Face dictionary from MTCNN with keypoints.
            
        Returns:
            numpy.ndarray: Aligned face image.
        """
        # Get facial landmarks
        left_eye, right_eye = self.get_eye_centers(face)
        
        # Calculate angle for rotation
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Get face box
        x, y, w, h = face['box']
        
        # Calculate center of face box
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        
        # Apply rotation
        rotated_image = cv2.warpAffine(
            image, 
            rotation_matrix, 
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # Get bounding box in rotated image
        # The box coordinates need to be transformed using the rotation matrix
        corners = np.array([
            [x, y],
            [x + w, y],
            [x, y + h],
            [x + w, y + h]
        ], dtype=np.float32)
        
        # Transform corners
        corners_transformed = np.zeros_like(corners)
        for i in range(4):
            corners_transformed[i, 0] = corners[i, 0] * rotation_matrix[0, 0] + \
                                      corners[i, 1] * rotation_matrix[0, 1] + \
                                      rotation_matrix[0, 2]
            corners_transformed[i, 1] = corners[i, 0] * rotation_matrix[1, 0] + \
                                      corners[i, 1] * rotation_matrix[1, 1] + \
                                      rotation_matrix[1, 2]
        
        # Get new bounding box
        x1 = np.min(corners_transformed[:, 0])
        y1 = np.min(corners_transformed[:, 1])
        x2 = np.max(corners_transformed[:, 0])
        y2 = np.max(corners_transformed[:, 1])
        
        # Add margin (20%)
        margin_x = int((x2 - x1) * 0.2)
        margin_y = int((y2 - y1) * 0.2)
        
        x1 = max(0, int(x1 - margin_x))
        y1 = max(0, int(y1 - margin_y))
        x2 = min(rotated_image.shape[1], int(x2 + margin_x))
        y2 = min(rotated_image.shape[0], int(y2 + margin_y))
        
        # Crop aligned face
        aligned_face = rotated_image[y1:y2, x1:x2]
        
        return aligned_face
    
    def crop_and_resize_face(self, image, face):
        """
        Crop and resize a face to the target size.
        
        Args:
            image (numpy.ndarray): Original image.
            face (dict): Face dictionary from MTCNN.
            
        Returns:
            numpy.ndarray: Cropped and resized face.
        """
        # First align the face
        aligned_face = self.align_face(image, face)
        
        # Resize to target size
        if aligned_face.shape[0] > 0 and aligned_face.shape[1] > 0:
            resized_face = cv2.resize(aligned_face, self.target_size, interpolation=cv2.INTER_CUBIC)
            return resized_face
        else:
            # If alignment failed, just crop and resize directly
            x, y, w, h = face['box']
            
            # Add margin (20%)
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(image.shape[1], x + w + margin_x)
            y2 = min(image.shape[0], y + h + margin_y)
            
            # Crop face
            cropped_face = image[y1:y2, x1:x2]
            
            # Resize
            if cropped_face.shape[0] > 0 and cropped_face.shape[1] > 0:
                resized_face = cv2.resize(cropped_face, self.target_size, interpolation=cv2.INTER_CUBIC)
                return resized_face
            
        return None
    
    def process_image(self, image, return_largest=True):
        """
        Process an image to detect, align, and crop faces.
        
        Args:
            image (numpy.ndarray): Image to process.
            return_largest (bool): Whether to return only the largest face or all faces.
            
        Returns:
            tuple: Processed face(s) and face detection metadata.
        """
        # Convert to RGB if needed
        if image is None:
            return None, {'status': 'failure', 'error': 'Image is None'}
        
        # Detect faces
        faces = self.detect_faces(image)
        
        if not faces:
            return None, {'status': 'no_face', 'count': 0}
        
        # For multiple faces, find the largest one or return all
        if return_largest:
            # Find the face with the largest area
            largest_face = max(faces, key=lambda face: face['box'][2] * face['box'][3])
            
            # Process the largest face
            processed_face = self.crop_and_resize_face(image, largest_face)
            
            if processed_face is not None:
                return processed_face, {
                    'status': 'success', 
                    'count': 1,
                    'confidence': largest_face['confidence'],
                    'box': largest_face['box']
                }
            else:
                return None, {'status': 'processing_failed', 'count': 1}
        else:
            # Process all faces
            processed_faces = []
            face_meta = []
            
            for face in faces:
                processed_face = self.crop_and_resize_face(image, face)
                if processed_face is not None:
                    processed_faces.append(processed_face)
                    face_meta.append({
                        'confidence': face['confidence'],
                        'box': face['box']
                    })
            
            if processed_faces:
                return processed_faces, {
                    'status': 'success', 
                    'count': len(processed_faces),
                    'faces': face_meta
                }
            else:
                return None, {'status': 'processing_failed', 'count': len(faces)}


def process_frame(frame_path, output_path, confidence_threshold=0.9, min_face_size=80, target_size=(224, 224), return_largest=True):
    """
    Process a single frame to detect and align faces.

    Args:
        frame_path (str): Path to the input frame.
        output_path (str): Path to save the processed face.
        confidence_threshold (float): Minimum confidence for valid face detections.
        min_face_size (int): Minimum width/height for valid faces.
        target_size (tuple): Size to resize faces to (width, height).
        return_largest (bool): Whether to return only the largest face if multiple are detected.

    Returns:
        dict: Processing statistics.
    """
    stats = {
        'frame': os.path.basename(frame_path),
        'status': 'failure',
        'error': None,
        'face_count': 0,
        'processing_time': 0
    }

    try:
        # Start timing
        start_time = time.time()

        # Read image
        image = cv2.imread(frame_path)
        if image is None:
            stats['error'] = "Failed to read image"
            return stats

        # Initialize face processor
        processor = FaceProcessor(
            confidence_threshold=confidence_threshold,
            min_face_size=min_face_size,
            target_size=target_size
        )

        # Process image
        faces, face_meta = processor.process_image(image, return_largest=return_largest)

        # Update stats
        stats['status'] = face_meta['status']
        stats['face_count'] = face_meta.get('count', 0)

        # Save processed faces
        if faces is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            if return_largest:
                # Save single face
                cv2.imwrite(output_path, faces[0])
            else:
                # Save all faces
                for i, face in enumerate(faces):
                    face_path = output_path.replace('.jpg', f'_face_{i + 1}.jpg')
                    cv2.imwrite(face_path, face)

        # End timing
        stats['processing_time'] = time.time() - start_time

    except Exception as e:
        stats['error'] = str(e)

    return stats


def process_frames_directory(input_dir, output_dir, confidence_threshold=0.9, min_face_size=80, 
                           target_size=(224, 224), return_largest=True, num_workers=4, 
                           recursive=False, logger=None):
    """
    Process all frames in a directory to detect and align faces.

    Args:
        input_dir (str): Directory containing input frames.
        output_dir (str): Directory to save processed faces.
        confidence_threshold (float): Minimum confidence for valid face detections.
        min_face_size (int): Minimum width/height for valid faces.
        target_size (tuple): Size to resize faces to (width, height).
        return_largest (bool): Whether to return only the largest face if multiple are detected.
        num_workers (int): Number of parallel workers.
        recursive (bool): Whether to process subdirectories recursively.
        logger (logging.Logger, optional): Logger object.

    Returns:
        dict: Processing statistics.
    """
    if logger is None:
        logger = logging.getLogger("FaceDetector")

    start_time = time.time()

    # Get all frame files
    frame_files = []
    if recursive:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    frame_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(input_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                frame_files.append(os.path.join(input_dir, file))

    logger.info(f"Found {len(frame_files)} frame files in {input_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process frames in parallel
    successful_frames = 0
    failed_frames = 0
    no_face_frames = 0
    total_faces = 0
    processing_results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks
        future_to_frame = {}
        for frame_file in frame_files:
            # Create output path that maintains subdirectory structure
            rel_path = os.path.relpath(frame_file, input_dir)
            rel_dir = os.path.dirname(rel_path)
            frame_name = os.path.basename(frame_file)

            # Replace extension with .jpg
            frame_name_base = os.path.splitext(frame_name)[0] + '.jpg'

            output_subdir = os.path.join(output_dir, rel_dir) if recursive else output_dir
            os.makedirs(output_subdir, exist_ok=True)
            output_path = os.path.join(output_subdir, frame_name_base)

            future = executor.submit(
                process_frame,
                frame_file,
                output_path,
                confidence_threshold,
                min_face_size,
                target_size,
                return_largest
            )
            future_to_frame[future] = frame_file

        # Process results as they complete
        for future in tqdm(as_completed(future_to_frame), total=len(frame_files), desc="Processing faces"):
            frame_file = future_to_frame[future]
            try:
                result = future.result()
                processing_results.append(result)

                if result['status'] == 'success':
                    successful_frames += 1
                    total_faces += result['face_count']
                    logger.debug(f"Successfully processed {os.path.basename(frame_file)}")
                elif result['status'] == 'no_face':
                    no_face_frames += 1
                    logger.debug(f"No face detected in {os.path.basename(frame_file)}")
                else:
                    failed_frames += 1
                    logger.warning(f"Failed to process {os.path.basename(frame_file)}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Error processing {os.path.basename(frame_file)}: {str(e)}")
                failed_frames += 1

    elapsed_time = time.time() - start_time
    logger.info(f"Face detection completed in {elapsed_time:.2f} seconds")
    logger.info(f"Total frames processed: {len(frame_files)}")
    logger.info(f"Successfully processed: {successful_frames} ({successful_frames / len(frame_files) * 100:.2f}%)")
    logger.info(f"No face detected: {no_face_frames} ({no_face_frames / len(frame_files) * 100:.2f}%)")
    logger.info(f"Failed to process: {failed_frames} ({failed_frames / len(frame_files) * 100:.2f}%)")
    logger.info(f"Total faces detected: {total_faces}")

    return {
        'total_frames': len(frame_files),
        'successful_frames': successful_frames,
        'failed_frames': failed_frames,
        'no_face_frames': no_face_frames,
        'total_faces': total_faces,
        'processing_time': elapsed_time,
        'results': processing_results
    }


def visualize_detection_results(results, output_file=None):
    """
    Visualize face detection results.
    
    Args:
        results (dict): Results from process_frames_directory
        output_file (str, optional): Path to save the visualization
    """
    # Create figure and subplots
    plt.figure(figsize=(12, 10))
    
    # Plot success rates
    plt.subplot(2, 2, 1)
    labels = ['Success', 'No Face', 'Failed']
    sizes = [
        results['successful_frames'],
        results['no_face_frames'],
        results['failed_frames']
    ]
    colors = ['#4CAF50', '#FFC107', '#F44336']
    explode = (0.1, 0, 0)
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.title('Face Detection Results')
    
    # Plot processing time histogram if available
    if 'processing_results' in results:
        processing_times = [r['processing_time'] for r in results['processing_results'] if 'processing_time' in r]
        if processing_times:
            plt.subplot(2, 2, 2)
            plt.hist(processing_times, bins=30, color='#2196F3', alpha=0.7)
            plt.xlabel('Processing Time (s)')
            plt.ylabel('Number of Frames')
            plt.title('Processing Time Distribution')
    
    # Display statistics
    plt.subplot(2, 2, 3)
    plt.axis('off')
    stats_text = (
        f"Total Frames: {results['total_frames']}\n"
        f"Successfully Processed: {results['successful_frames']} ({results['success_rate']:.2f}%)\n"
        f"No Face Detected: {results['no_face_frames']} ({results['no_face_rate']:.2f}%)\n"
        f"Failed to Process: {results['failed_frames']} ({results['failure_rate']:.2f}%)\n"
        f"Total Faces Detected: {results['total_faces']}\n"
        f"Total Processing Time: {results['processing_time']:.2f} seconds\n"
        f"Average Processing Time: {results['processing_time']/results['total_frames']:.4f} seconds/frame"
    )
    plt.text(0.1, 0.5, stats_text, fontsize=12)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Detect and align faces in video frames")
    
    # Input/output options
    parser.add_argument('--input-dir', type=str, required=True,
                        help="Directory containing input frames")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Directory to save processed faces")
    
    # Processing options
    parser.add_argument('--confidence', type=float, default=0.9,
                        help="Minimum face detection confidence threshold (0-1)")
    parser.add_argument('--min-size', type=int, default=80,
                        help="Minimum face size to process (pixels)")
    parser.add_argument('--target-size', type=str, default="224x224",
                        help="Target size for face images (WxH)")
    parser.add_argument('--all-faces', action='store_true',
                        help="Process all detected faces instead of just the largest")
    parser.add_argument('--recursive', action='store_true',
                        help="Process subdirectories recursively")
    
    # Processing options
    parser.add_argument('--workers', type=int, default=4,
                        help="Number of parallel workers")
    
    # Visualization options
    parser.add_argument('--visualize', action='store_true',
                        help="Visualize detection results")
    parser.add_argument('--vis-output', type=str,
                        help="Path to save visualization")
    
    # Logging options
    parser.add_argument('--log-file', type=str,
                        help="Path to log file")
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Set the logging level")
    
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger(args.log_file)
    logger.setLevel(getattr(logging, args.log_level))
    
    # Parse target size
    try:
        target_w, target_h = map(int, args.target_size.split('x'))
        target_size = (target_w, target_h)
    except ValueError:
        logger.error("Invalid target size format. Use WxH (e.g., 224x224)")
        return
    
    # Process frames
    logger.info(f"Starting face detection and alignment on frames in {args.input_dir}")
    logger.info(f"Confidence threshold: {args.confidence}")
    logger.info(f"Minimum face size: {args.min_size}px")
    logger.info(f"Target size: {target_size[0]}x{target_size[1]}px")
    logger.info(f"Processing {'all faces' if args.all_faces else 'largest face only'}")
    
    results = process_frames_directory(
        args.input_dir,
        args.output_dir,
        args.confidence,
        args.min_size,
        target_size,
        not args.all_faces,
        args.workers,
        args.recursive,
        logger
    )
    
    # Visualize results if requested
    if args.visualize:
        vis_output = args.vis_output or os.path.join(args.output_dir, "detection_results.png")
        logger.info(f"Generating visualization at {vis_output}")
        visualize_detection_results(results, vis_output)

if __name__ == "__main__":
    main()
