#!/usr/bin/env python3
"""
GPU-Accelerated Preprocessing Pipeline for Cross-Attention CNN
Extracts frames, faces, optical flow, and features with GPU acceleration
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import logging
from pathlib import Path
import time
from typing import List, Tuple, Optional
import json
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/gpu_preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GPUPreprocessingPipeline:
    """Main preprocessing pipeline with GPU acceleration"""
    
    def __init__(self):
        self.setup_directories()
        self.check_gpu()
        self.detector = MTCNN()
        self.target_face_size = (224, 224)
        self.frames_per_video = 80
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            'data/processed/frames',
            'data/processed/faces', 
            'data/processed/optical_flow',
            'logs'
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
    def check_gpu(self):
        """Check GPU availability"""
        gpus = tf.config.experimental.list_logical_devices('GPU')
        logger.info(f"GPU devices: {gpus}")
        if gpus:
            logger.info("✅ GPU acceleration available")
            # Configure GPU memory growth
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            if physical_devices:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
        else:
            logger.warning("⚠️ No GPU found, using CPU")
            
    def extract_frames_from_video(self, video_path: str, output_dir: str, max_frames: int = 80) -> int:
        """Extract frames from a single video"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return 0
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame indices to extract evenly
            if total_frames <= max_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
            
            extracted_count = 0
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    frame_filename = f"frame_{i:04d}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    extracted_count += 1
                    
            cap.release()
            logger.info(f"Extracted {extracted_count} frames from {os.path.basename(video_path)}")
            return extracted_count
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            return 0
    
    def extract_frames_from_all_videos(self):
        """Extract frames from all videos in the dataset"""
        logger.info("🎬 Starting frame extraction...")
        
        video_base_dir = 'data/raw/train-1'
        frames_base_dir = 'data/processed/frames'
        
        total_extracted = 0
        failed_videos = []
        
        # Get all video directories
        video_dirs = [d for d in os.listdir(video_base_dir) 
                     if os.path.isdir(os.path.join(video_base_dir, d))]
        
        for video_dir in tqdm(sorted(video_dirs), desc="Processing video directories"):
            video_dir_path = os.path.join(video_base_dir, video_dir)
            frames_dir_path = os.path.join(frames_base_dir, video_dir)
            
            # Create frames directory
            Path(frames_dir_path).mkdir(parents=True, exist_ok=True)
            
            # Get all video files
            video_files = [f for f in os.listdir(video_dir_path) if f.endswith('.mp4')]
            
            for video_file in video_files:
                video_path = os.path.join(video_dir_path, video_file)
                
                # Create subdirectory for this video's frames
                video_name = os.path.splitext(video_file)[0]
                video_frames_dir = os.path.join(frames_dir_path, video_name)
                Path(video_frames_dir).mkdir(parents=True, exist_ok=True)
                
                extracted = self.extract_frames_from_video(video_path, video_frames_dir)
                if extracted > 0:
                    total_extracted += extracted
                else:
                    failed_videos.append(video_path)
        
        logger.info(f"✅ Frame extraction complete. Total frames: {total_extracted}")
        if failed_videos:
            logger.warning(f"⚠️ Failed videos: {len(failed_videos)}")
            
        return total_extracted, failed_videos
    
    def extract_face_from_frame(self, frame_path: str) -> Optional[np.ndarray]:
        """Extract face from a single frame using MTCNN"""
        try:
            # Read image
            image = cv2.imread(frame_path)
            if image is None:
                return None
                
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            result = self.detector.detect_faces(rgb_image)
            
            if result:
                # Use the first (most confident) face
                face = result[0]
                x, y, width, height = face['box']
                
                # Add padding and ensure bounds
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                width = min(rgb_image.shape[1] - x, width + 2*padding)
                height = min(rgb_image.shape[0] - y, height + 2*padding)
                
                # Extract face region
                face_img = rgb_image[y:y+height, x:x+width]
                
                # Resize to target size
                face_resized = cv2.resize(face_img, self.target_face_size)
                
                # Convert back to BGR for saving
                face_bgr = cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)
                
                return face_bgr
                
        except Exception as e:
            logger.error(f"Error extracting face from {frame_path}: {e}")
            
        return None
    
    def extract_faces_from_directory(self, frames_dir: str, faces_dir: str) -> int:
        """Extract faces from all frames in a directory"""
        Path(faces_dir).mkdir(parents=True, exist_ok=True)
        
        frame_files = [f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))]
        extracted_count = 0
        
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            face_img = self.extract_face_from_frame(frame_path)
            
            if face_img is not None:
                face_filename = f"face_{os.path.splitext(frame_file)[0]}.jpg"
                face_path = os.path.join(faces_dir, face_filename)
                cv2.imwrite(face_path, face_img)
                extracted_count += 1
                
        return extracted_count
    
    def extract_faces_from_all_frames(self):
        """Extract faces from all frame directories"""
        logger.info("😊 Starting face extraction with GPU-accelerated MTCNN...")
        
        frames_base_dir = 'data/processed/frames'
        faces_base_dir = 'data/processed/faces'
        
        total_faces = 0
        failed_dirs = []
        
        # Get all frame directories
        frame_dirs = [d for d in os.listdir(frames_base_dir) 
                     if os.path.isdir(os.path.join(frames_base_dir, d))]
        
        for frame_dir in tqdm(sorted(frame_dirs), desc="Extracting faces"):
            frame_dir_path = os.path.join(frames_base_dir, frame_dir)
            faces_dir_path = os.path.join(faces_base_dir, frame_dir)
            
            # Process each video subdirectory
            video_subdirs = [d for d in os.listdir(frame_dir_path) 
                           if os.path.isdir(os.path.join(frame_dir_path, d))]
            
            for video_subdir in video_subdirs:
                video_frames_path = os.path.join(frame_dir_path, video_subdir)
                video_faces_path = os.path.join(faces_dir_path, video_subdir)
                
                extracted = self.extract_faces_from_directory(video_frames_path, video_faces_path)
                if extracted > 0:
                    total_faces += extracted
                    logger.info(f"Extracted {extracted} faces from {frame_dir}/{video_subdir}")
                else:
                    failed_dirs.append(f"{frame_dir}/{video_subdir}")
        
        logger.info(f"✅ Face extraction complete. Total faces: {total_faces}")
        if failed_dirs:
            logger.warning(f"⚠️ Failed directories: {len(failed_dirs)}")
            
        return total_faces, failed_dirs
    
    def compute_optical_flow_sequence(self, face_dir: str, output_dir: str) -> int:
        """Compute optical flow for a sequence of face images"""
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Get all face images
            face_files = sorted([f for f in os.listdir(face_dir) if f.endswith('.jpg')])
            
            if len(face_files) < 2:
                logger.warning(f"Not enough faces for optical flow in {face_dir}")
                return 0
            
            flow_count = 0
            
            for i in range(len(face_files) - 1):
                # Read consecutive frames
                frame1_path = os.path.join(face_dir, face_files[i])
                frame2_path = os.path.join(face_dir, face_files[i + 1])
                
                frame1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
                frame2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)
                
                if frame1 is not None and frame2 is not None:
                    # Compute optical flow using Farneback method
                    flow = cv2.calcOpticalFlowPyrLK(
                        frame1, frame2, None, None,
                        winSize=(15, 15),
                        maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                    )
                    
                    # Save optical flow as numpy array
                    flow_filename = f"flow_{i:04d}_{i+1:04d}.npy"
                    flow_path = os.path.join(output_dir, flow_filename)
                    np.save(flow_path, flow)
                    flow_count += 1
            
            return flow_count
            
        except Exception as e:
            logger.error(f"Error computing optical flow for {face_dir}: {e}")
            return 0
    
    def compute_all_optical_flow(self):
        """Compute optical flow for all face sequences"""
        logger.info("🌊 Starting optical flow computation...")
        
        faces_base_dir = 'data/processed/faces'
        flow_base_dir = 'data/processed/optical_flow'
        
        total_flows = 0
        failed_sequences = []
        
        # Get all face directories
        if not os.path.exists(faces_base_dir):
            logger.error("No faces directory found. Please run face extraction first.")
            return 0, []
        
        face_dirs = [d for d in os.listdir(faces_base_dir) 
                    if os.path.isdir(os.path.join(faces_base_dir, d))]
        
        for face_dir in tqdm(sorted(face_dirs), desc="Computing optical flow"):
            face_dir_path = os.path.join(faces_base_dir, face_dir)
            flow_dir_path = os.path.join(flow_base_dir, face_dir)
            
            # Process each video subdirectory
            video_subdirs = [d for d in os.listdir(face_dir_path) 
                           if os.path.isdir(os.path.join(face_dir_path, d))]
            
            for video_subdir in video_subdirs:
                video_faces_path = os.path.join(face_dir_path, video_subdir)
                video_flow_path = os.path.join(flow_dir_path, video_subdir)
                
                flow_count = self.compute_optical_flow_sequence(video_faces_path, video_flow_path)
                if flow_count > 0:
                    total_flows += flow_count
                    logger.info(f"Computed {flow_count} optical flows for {face_dir}/{video_subdir}")
                else:
                    failed_sequences.append(f"{face_dir}/{video_subdir}")
        
        logger.info(f"✅ Optical flow computation complete. Total flows: {total_flows}")
        if failed_sequences:
            logger.warning(f"⚠️ Failed sequences: {len(failed_sequences)}")
            
        return total_flows, failed_sequences
    
    def save_preprocessing_summary(self, results: dict):
        """Save preprocessing results summary"""
        summary_path = 'results/preprocessing_summary.json'
        Path('results').mkdir(exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump(results, indent=2, fp=f)
        
        logger.info(f"Preprocessing summary saved to {summary_path}")
    
    def run_complete_pipeline(self):
        """Run the complete preprocessing pipeline"""
        logger.info("🚀 Starting GPU-accelerated preprocessing pipeline...")
        start_time = time.time()
        
        results = {
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'gpu_available': len(tf.config.experimental.list_logical_devices('GPU')) > 0
        }
        
        try:
            # Step 1: Extract frames
            logger.info("\n" + "="*60)
            logger.info("STEP 1: FRAME EXTRACTION")
            logger.info("="*60)
            total_frames, failed_videos = self.extract_frames_from_all_videos()
            results['frames'] = {
                'total_extracted': total_frames,
                'failed_videos': len(failed_videos)
            }
            
            # Step 2: Extract faces
            logger.info("\n" + "="*60)
            logger.info("STEP 2: FACE EXTRACTION")
            logger.info("="*60)
            total_faces, failed_face_dirs = self.extract_faces_from_all_frames()
            results['faces'] = {
                'total_extracted': total_faces,
                'failed_directories': len(failed_face_dirs)
            }
            
            # Step 3: Compute optical flow
            logger.info("\n" + "="*60)
            logger.info("STEP 3: OPTICAL FLOW COMPUTATION")
            logger.info("="*60)
            total_flows, failed_flow_seqs = self.compute_all_optical_flow()
            results['optical_flow'] = {
                'total_computed': total_flows,
                'failed_sequences': len(failed_flow_seqs)
            }
            
            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time
            results['processing_time_seconds'] = processing_time
            results['processing_time_formatted'] = str(time.timedelta(seconds=int(processing_time)))
            results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
            results['status'] = 'completed'
            
            # Save summary
            self.save_preprocessing_summary(results)
            
            logger.info("\n" + "="*60)
            logger.info("🎉 PREPROCESSING PIPELINE COMPLETED!")
            logger.info("="*60)
            logger.info(f"⏱️  Total processing time: {results['processing_time_formatted']}")
            logger.info(f"🎬 Frames extracted: {total_frames}")
            logger.info(f"😊 Faces extracted: {total_faces}")
            logger.info(f"🌊 Optical flows computed: {total_flows}")
            logger.info(f"🎯 GPU acceleration: {'✅ Enabled' if results['gpu_available'] else '❌ Disabled'}")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
            self.save_preprocessing_summary(results)
            raise

def main():
    """Main function to run the preprocessing pipeline"""
    pipeline = GPUPreprocessingPipeline()
    results = pipeline.run_complete_pipeline()
    
    print("\n🎉 Preprocessing pipeline completed successfully!")
    print(f"Check 'results/preprocessing_summary.json' for detailed results.")

if __name__ == "__main__":
    main()
