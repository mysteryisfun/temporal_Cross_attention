# Frame extraction batch script for Cross-Attention CNN Research Project
# This script extracts frames from all videos in the train-1 directory

# Check if virtual environment exists and activate it
if (Test-Path .\env\Scripts\activate.ps1) {
    # Activate the virtual environment
    .\env\Scripts\activate.ps1

    # Create output directory
    $outputDir = "data/processed/frames"
    if (-not (Test-Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir -Force
        Write-Host "Created output directory: $outputDir"
    } else {
        Write-Host "Output directory already exists: $outputDir"
    }

    # Extract frames from videos with 5 fps sampling rate (standard for the project)
    Write-Host "Starting frame extraction at 5 fps..."
    python scripts/extract_frames.py --video-dir data/raw/train-1 --output-dir $outputDir --sampling-rate 5 --resize 224x224

    Write-Host "Frame extraction completed!"
    
    # Deactivate virtual environment if needed
    # deactivate
} else {
    Write-Host "Virtual environment not found. Please create and set up the virtual environment first."
    exit 1
}
