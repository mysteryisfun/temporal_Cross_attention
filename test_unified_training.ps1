# PowerShell script to test the unified training system
# Author: [Your Name]
# Date: May 20, 2025

# Check for Python environment
if (Test-Path .\env\Scripts\activate.ps1) {
    Write-Host "Activating virtual environment..." -ForegroundColor Green
    .\env\Scripts\activate.ps1
    
    # Create output directories if they don't exist
    $outputDir = "results/test_phase_4.3/unified_training"
    if (-not (Test-Path $outputDir)) {
        Write-Host "Creating output directory: $outputDir" -ForegroundColor Yellow
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    }
    
    # Run the unified training system test
    Write-Host "Running unified training system test..." -ForegroundColor Cyan
    python scripts/test_unified_training.py `
        --static_features data_sample/static_features.npy `
        --dynamic_features data_sample/dynamic_features.npy `
        --labels data_sample/labels.npy `
        --output results/test_phase_4.3/unified_training/model.h5 `
        --epochs 3 `
        --batch_size 4
    
    # Check if training completed successfully
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Unified training system test completed successfully!" -ForegroundColor Green
        Write-Host "Results saved to: $outputDir" -ForegroundColor Green
    } else {
        Write-Host "Unified training system test failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    }
    
    # Deactivate virtual environment
    deactivate
} else {
    Write-Host "Python virtual environment not found. Please create it first." -ForegroundColor Red
    Write-Host "Try running: python -m venv env" -ForegroundColor Yellow
    exit 1
}
