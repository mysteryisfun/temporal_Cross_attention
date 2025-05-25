#!/usr/bin/env python3
"""
GPU and Dependencies Check for Face Extraction
"""
import sys

def check_tensorflow_gpu():
    """Check TensorFlow GPU setup"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Check for GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        print(f"🔍 GPU devices found: {len(gpus)}")
        
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
            
            # Check if GPU is actually available for computation
            print(f"🔧 CUDA built: {tf.test.is_built_with_cuda()}")
            
            # Try to create a simple operation on GPU
            try:
                with tf.device('/GPU:0'):
                    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                    c = tf.matmul(a, b)
                print("✅ GPU computation test: PASSED")
                return True
            except Exception as e:
                print(f"❌ GPU computation test failed: {e}")
                return False
        else:
            print("❌ No GPU devices found")
            return False
            
    except ImportError:
        print("❌ TensorFlow not available")
        return False

def check_pytorch_gpu():
    """Check PyTorch GPU setup"""
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.is_available()}")
            print(f"🔍 GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("❌ CUDA not available in PyTorch")
            return False
            
    except ImportError:
        print("❌ PyTorch not available")
        return False

def check_mtcnn():
    """Check MTCNN availability"""
    try:
        import mtcnn
        print(f"✅ MTCNN available: {mtcnn.__version__}")
        
        # Try to create MTCNN detector
        from mtcnn import MTCNN
        detector = MTCNN()
        print("✅ MTCNN detector created successfully")
        return True
        
    except ImportError:
        print("❌ MTCNN not available")
        return False
    except Exception as e:
        print(f"❌ MTCNN error: {e}")
        return False

def check_opencv():
    """Check OpenCV availability"""
    try:
        import cv2
        print(f"✅ OpenCV available: {cv2.__version__}")
        
        # Check if OpenCV was built with CUDA support
        build_info = cv2.getBuildInformation()
        if "CUDA" in build_info:
            print("✅ OpenCV built with CUDA support")
            return True
        else:
            print("⚠️ OpenCV built without CUDA support")
            return False
            
    except ImportError:
        print("❌ OpenCV not available")
        return False

def check_gpu_memory():
    """Check available GPU memory"""
    try:
        import tensorflow as tf
        
        if tf.config.list_physical_devices('GPU'):
            # Get GPU memory info
            gpu_details = tf.config.experimental.get_device_details(
                tf.config.list_physical_devices('GPU')[0]
            )
            if 'device_name' in gpu_details:
                print(f"🔧 GPU Device: {gpu_details['device_name']}")
            
            # Try to get memory info
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    memory_info = result.stdout.strip().split('\n')[0].split(', ')
                    total_memory = int(memory_info[0])
                    free_memory = int(memory_info[1])
                    print(f"🔧 GPU Memory - Total: {total_memory}MB, Free: {free_memory}MB")
                    return total_memory, free_memory
            except:
                print("⚠️ Could not get detailed GPU memory info")
                
    except Exception as e:
        print(f"⚠️ GPU memory check failed: {e}")
    
    return None, None

def main():
    """Main function to run all checks"""
    print("🔍 GPU and Dependencies Check for Face Extraction")
    print("=" * 60)
    
    # Check TensorFlow GPU
    print("\n📦 TensorFlow GPU Check:")
    tf_gpu = check_tensorflow_gpu()
    
    # Check PyTorch GPU  
    print("\n📦 PyTorch GPU Check:")
    pt_gpu = check_pytorch_gpu()
    
    # Check MTCNN
    print("\n📦 MTCNN Check:")
    mtcnn_available = check_mtcnn()
    
    # Check OpenCV
    print("\n📦 OpenCV Check:")
    cv_available = check_opencv()
    
    # Check GPU memory
    print("\n📦 GPU Memory Check:")
    total_mem, free_mem = check_gpu_memory()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    print(f"   TensorFlow GPU: {'✅ Available' if tf_gpu else '❌ Not Available'}")
    print(f"   PyTorch GPU: {'✅ Available' if pt_gpu else '❌ Not Available'}")
    print(f"   MTCNN: {'✅ Available' if mtcnn_available else '❌ Not Available'}")
    print(f"   OpenCV: {'✅ Available' if cv_available else '❌ Not Available'}")
    
    if tf_gpu or pt_gpu:
        print("\n🚀 RECOMMENDATION: GPU acceleration available for face extraction!")
        if mtcnn_available:
            print("   MTCNN can use GPU acceleration for faster face detection.")
        else:
            print("   Install MTCNN: pip install mtcnn")
    else:
        print("\n⚠️ RECOMMENDATION: GPU not available. Face extraction will run on CPU.")
        print("   This will be slower but still functional.")
        
    if not cv_available:
        print("   Install OpenCV: pip install opencv-python")

if __name__ == "__main__":
    main()
