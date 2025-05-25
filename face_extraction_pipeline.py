#!/usr/bin/env python3
"""
Face Extraction and Optical Flow Pipeline for Cross-Attention CNN
Continues from already extracted frames using GPU-accelerated MTCNN
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
        logging.FileHandler('logs/face_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FaceExtractionPipeline:
    """Face extraction and optical flow pipeline with GPU acceleration"""
    
    def __init__(self):
        self.setup_directories()
        self.check_gpu()
        self.detector = MTCNN()
        self.target_face_size = (224, 224)
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            'data/processed/faces', 
            'data/processed/optical_flow',
            'logs',
            'results'
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
    
    def extract_faces_from_video_directory(self, video_frames_dir: str, video_faces_dir: str) -> int:
        """Extract faces from all frames in a single video directory"""
        Path(video_faces_dir).mkdir(parents=True, exist_ok=True)
        
        frame_files = sorted([f for f in os.listdir(video_frames_dir) if f.endswith('.jpg')])
        extracted_count = 0
        
        for frame_file in frame_files:
            frame_path = os.path.join(video_frames_dir, frame_file)
            face_img = self.extract_face_from_frame(frame_path)
            
            if face_img is not None:
                # Keep same filename but change prefix to face_
                face_filename = frame_file.replace('frame_', 'face_')
                face_path = os.path.join(video_faces_dir, face_filename)
                cv2.imwrite(face_path, face_img)
                extracted_count += 1
                
        return extracted_count
    
    def extract_all_faces(self):
        """Extract faces from all frame directories"""
        logger.info("😊 Starting face extraction with GPU-accelerated MTCNN...")
        
        frames_base_dir = 'data/processed/frames'
        faces_base_dir = 'data/processed/faces'
        
        total_faces = 0
        failed_videos = []
        processing_stats = {}
        
        # Get all training directories
        training_dirs = sorted([d for d in os.listdir(frames_base_dir) 
                               if os.path.isdir(os.path.join(frames_base_dir, d))])
        
        logger.info(f"Found {len(training_dirs)} training directories")
        
        for training_dir in tqdm(training_dirs, desc="Processing training directories"):
            training_frames_path = os.path.join(frames_base_dir, training_dir)
            training_faces_path = os.path.join(faces_base_dir, training_dir)
            
            # Get all video directories in this training directory
            video_dirs = sorted([d for d in os.listdir(training_frames_path) 
                               if os.path.isdir(os.path.join(training_frames_path, d))])
            
            training_faces = 0
            training_failed = 0
            
            for video_dir in tqdm(video_dirs, desc=f"Videos in {training_dir}", leave=False):
                video_frames_path = os.path.join(training_frames_path, video_dir)
                video_faces_path = os.path.join(training_faces_path, video_dir)
                
                extracted = self.extract_faces_from_video_directory(video_frames_path, video_faces_path)
                
                if extracted > 0:
                    total_faces += extracted
                    training_faces += extracted
                else:
                    failed_videos.append(f"{training_dir}/{video_dir}")
                    training_failed += 1
            
            processing_stats[training_dir] = {
                'videos_processed': len(video_dirs),
                'faces_extracted': training_faces,
                'failed_videos': training_failed
            }
            
            logger.info(f"✅ {training_dir}: {training_faces} faces from {len(video_dirs)} videos")
        
        logger.info(f"🎉 Face extraction complete!")
        logger.info(f"📊 Total faces extracted: {total_faces}")
        logger.info(f"❌ Failed videos: {len(failed_videos)}")
        
        return total_faces, failed_videos, processing_stats
    
    def compute_optical_flow_for_video(self, faces_dir: str, flow_dir: str) -> int:
        """Compute optical flow for a sequence of face images in one video"""
        try:
            Path(flow_dir).mkdir(parents=True, exist_ok=True)
            
            # Get all face images
            face_files = sorted([f for f in os.listdir(faces_dir) if f.endswith('.jpg')])
            
            if len(face_files) < 2:
                logger.warning(f"Not enough faces for optical flow in {faces_dir}")
                return 0
            
            flow_count = 0
            
            for i in range(len(face_files) - 1):
                # Read consecutive frames
                frame1_path = os.path.join(faces_dir, face_files[i])
                frame2_path = os.path.join(faces_dir, face_files[i + 1])
                
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
                    flow_path = os.path.join(flow_dir, flow_filename)
                    np.save(flow_path, flow)
                    flow_count += 1
            
            return flow_count
            
        except Exception as e:
            logger.error(f"Error computing optical flow for {faces_dir}: {e}")
            return 0
    
    def compute_all_optical_flow(self):
        """Compute optical flow for all face sequences"""
        logger.info("🌊 Starting optical flow computation...")
        
        faces_base_dir = 'data/processed/faces'
        flow_base_dir = 'data/processed/optical_flow'
        
        if not os.path.exists(faces_base_dir):
            logger.error("No faces directory found. Please run face extraction first.")
            return 0, [], {}
        
        total_flows = 0
        failed_videos = []
        processing_stats = {}
        
        # Get all training directories
        training_dirs = sorted([d for d in os.listdir(faces_base_dir) 
                               if os.path.isdir(os.path.join(faces_base_dir, d))])
        
        logger.info(f"Found {len(training_dirs)} training directories with faces")
        
        for training_dir in tqdm(training_dirs, desc="Computing optical flow"):
            training_faces_path = os.path.join(faces_base_dir, training_dir)
            training_flow_path = os.path.join(flow_base_dir, training_dir)
            
            # Get all video directories
            video_dirs = sorted([d for d in os.listdir(training_faces_path) 
                               if os.path.isdir(os.path.join(training_faces_path, d))])
            
            training_flows = 0
            training_failed = 0
            
            for video_dir in tqdm(video_dirs, desc=f"Optical flow for {training_dir}", leave=False):
                video_faces_path = os.path.join(training_faces_path, video_dir)
                video_flow_path = os.path.join(training_flow_path, video_dir)
                
                flow_count = self.compute_optical_flow_for_video(video_faces_path, video_flow_path)
                
                if flow_count > 0:
                    total_flows += flow_count
                    training_flows += flow_count
                else:
                    failed_videos.append(f"{training_dir}/{video_dir}")
                    training_failed += 1
            
            processing_stats[training_dir] = {
                'videos_processed': len(video_dirs),
                'flows_computed': training_flows,
                'failed_videos': training_failed
            }
            
            logger.info(f"✅ {training_dir}: {training_flows} optical flows from {len(video_dirs)} videos")
        
        logger.info(f"🎉 Optical flow computation complete!")
        logger.info(f"📊 Total optical flows: {total_flows}")
        logger.info(f"❌ Failed videos: {len(failed_videos)}")
        
        return total_flows, failed_videos, processing_stats
    
    def verify_data_structure(self):
        """Verify the final data structure"""
        logger.info("🔍 Verifying data structure...")
        
        # Check frames
        frames_dir = 'data/processed/frames'
        faces_dir = 'data/processed/faces'
        flow_dir = 'data/processed/optical_flow'
        labels_file = 'data/training_data/labels_train1.npy'
        
        verification = {
            'frames': {},
            'faces': {},
            'optical_flow': {},
            'labels': {}
        }
        
        # Check frames
        if os.path.exists(frames_dir):
            training_dirs = [d for d in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, d))]
            total_videos = 0
            total_frames = 0
            
            for training_dir in training_dirs:
                training_path = os.path.join(frames_dir, training_dir)
                video_dirs = [d for d in os.listdir(training_path) if os.path.isdir(os.path.join(training_path, d))]
                
                dir_frames = 0
                for video_dir in video_dirs:
                    video_path = os.path.join(training_path, video_dir)
                    frame_files = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
                    dir_frames += len(frame_files)
                
                total_videos += len(video_dirs)
                total_frames += dir_frames
                verification['frames'][training_dir] = {'videos': len(video_dirs), 'frames': dir_frames}
            
            verification['frames']['total'] = {'videos': total_videos, 'frames': total_frames}
        
        # Check faces
        if os.path.exists(faces_dir):
            training_dirs = [d for d in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, d))]
            total_videos = 0
            total_faces = 0
            
            for training_dir in training_dirs:
                training_path = os.path.join(faces_dir, training_dir)
                video_dirs = [d for d in os.listdir(training_path) if os.path.isdir(os.path.join(training_path, d))]
                
                dir_faces = 0
                for video_dir in video_dirs:
                    video_path = os.path.join(training_path, video_dir)
                    face_files = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
                    dir_faces += len(face_files)
                
                total_videos += len(video_dirs)
                total_faces += dir_faces
                verification['faces'][training_dir] = {'videos': len(video_dirs), 'faces': dir_faces}
            
            verification['faces']['total'] = {'videos': total_videos, 'faces': total_faces}
        
        # Check optical flow
        if os.path.exists(flow_dir):
            training_dirs = [d for d in os.listdir(flow_dir) if os.path.isdir(os.path.join(flow_dir, d))]
            total_videos = 0
            total_flows = 0
            
            for training_dir in training_dirs:
                training_path = os.path.join(flow_dir, training_dir)
                video_dirs = [d for d in os.listdir(training_path) if os.path.isdir(os.path.join(training_path, d))]
                
                dir_flows = 0
                for video_dir in video_dirs:
                    video_path = os.path.join(training_path, video_dir)
                    flow_files = [f for f in os.listdir(video_path) if f.endswith('.npy')]
                    dir_flows += len(flow_files)
                
                total_videos += len(video_dirs)
                total_flows += dir_flows
                verification['optical_flow'][training_dir] = {'videos': len(video_dirs), 'flows': dir_flows}
            
            verification['optical_flow']['total'] = {'videos': total_videos, 'flows': total_flows}
        
        # Check labels
        if os.path.exists(labels_file):
            labels = np.load(labels_file)
            verification['labels'] = {
                'file_exists': True,
                'shape': labels.shape,
                'total_samples': labels.shape[0] if len(labels.shape) > 0 else 0
            }
        
        logger.info("📋 Data Structure Verification:")
        logger.info(f"  Frames: {verification['frames'].get('total', {}).get('videos', 0)} videos, {verification['frames'].get('total', {}).get('frames', 0)} frames")
        logger.info(f"  Faces: {verification['faces'].get('total', {}).get('videos', 0)} videos, {verification['faces'].get('total', {}).get('faces', 0)} faces")
        logger.info(f"  Optical Flow: {verification['optical_flow'].get('total', {}).get('videos', 0)} videos, {verification['optical_flow'].get('total', {}).get('flows', 0)} flows")
        logger.info(f"  Labels: {verification['labels'].get('total_samples', 0)} samples")
        
        return verification
    
    def save_processing_summary(self, results: dict):
        """Save processing results summary"""
        summary_path = 'results/face_extraction_summary.json'
        
        with open(summary_path, 'w') as f:
            json.dump(results, indent=2, fp=f)
        
        logger.info(f"Processing summary saved to {summary_path}")
    
    def run_pipeline(self):
        """Run the complete face extraction and optical flow pipeline"""
        logger.info("🚀 Starting Face Extraction and Optical Flow Pipeline...")
        start_time = time.time()
        
        results = {
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'gpu_available': len(tf.config.experimental.list_logical_devices('GPU')) > 0,
            'pipeline_version': 'face_extraction_v1.0'
        }
        
        try:
            # Step 1: Extract faces
            logger.info("\n" + "="*60)
            logger.info("STEP 1: FACE EXTRACTION")
            logger.info("="*60)
            total_faces, failed_face_videos, face_stats = self.extract_all_faces()
            results['face_extraction'] = {
                'total_faces': total_faces,
                'failed_videos': len(failed_face_videos),
                'processing_stats': face_stats
            }
            
            # Step 2: Compute optical flow
            logger.info("\n" + "="*60)
            logger.info("STEP 2: OPTICAL FLOW COMPUTATION")
            logger.info("="*60)
            total_flows, failed_flow_videos, flow_stats = self.compute_all_optical_flow()
            results['optical_flow'] = {
                'total_flows': total_flows,
                'failed_videos': len(failed_flow_videos),
                'processing_stats': flow_stats
            }
            
            # Step 3: Verify data structure
            logger.info("\n" + "="*60)
            logger.info("STEP 3: DATA STRUCTURE VERIFICATION")
            logger.info("="*60)
            verification = self.verify_data_structure()
            results['data_verification'] = verification
            
            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time
            results['processing_time_seconds'] = processing_time
            results['processing_time_formatted'] = str(time.timedelta(seconds=int(processing_time)))
            results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
            results['status'] = 'completed'
            
            # Save summary
            self.save_processing_summary(results)
            
            logger.info("\n" + "="*60)
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            logger.info(f"⏱️  Total processing time: {results['processing_time_formatted']}")
            logger.info(f"😊 Faces extracted: {total_faces}")
            logger.info(f"🌊 Optical flows computed: {total_flows}")
            logger.info(f"🎯 GPU acceleration: {'✅ Enabled' if results['gpu_available'] else '❌ Disabled'}")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
            self.save_processing_summary(results)
            raise

def main():
    """Main function to run the face extraction pipeline"""
    pipeline = FaceExtractionPipeline()
    results = pipeline.run_pipeline()
    
    print("\n🎉 Face extraction and optical flow pipeline completed!")
    print(f"Check 'results/face_extraction_summary.json' for detailed results.")

if __name__ == "__main__":
    main()
