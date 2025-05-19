# Script to generate visualizations from training logs for Phase 4.3
Write-Host "Cross-Attention CNN Research - Advanced Training Visualization Tool" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

param(
    [string]$BatchMetricsPath = "results/batch_metrics/metrics_history.json",
    [string]$TimeLogsPath = "results/time_tracking/time_logs.json",
    [string]$LRLogsPath = "results/lr_tracking/lr_logs.json",
    [string]$OutputDir = "results/training_visualizations",
    [string]$ModelPath = "results/personality_model_trained.h5",
    [string]$StaticFeaturesPath = "data/static_features.npy",
    [string]$DynamicFeaturesPath = "data/dynamic_features.npy",
    [switch]$RunAll = $false
)

# Check if virtual environment exists
if (Test-Path .\env\Scripts\activate.ps1) {
    # Activate the environment
    Write-Host "`nActivating virtual environment..." -ForegroundColor Green
    . .\env\Scripts\activate.ps1
    
    # Verify input files exist
    $inputFilesExist = $true
    if (-not (Test-Path $BatchMetricsPath)) {
        Write-Host "Metrics file not found: $BatchMetricsPath" -ForegroundColor Red
        $inputFilesExist = $false
    }
    if (-not (Test-Path $TimeLogsPath)) {
        Write-Host "Time logs file not found: $TimeLogsPath" -ForegroundColor Red
        $inputFilesExist = $false
    }
    if (-not (Test-Path $LRLogsPath)) {
        Write-Host "Learning rate logs file not found: $LRLogsPath" -ForegroundColor Red
        $inputFilesExist = $false
    }
    
    if (-not $inputFilesExist) {
        Write-Host "`nRequired input files missing. Make sure you've run training with Phase 4.3 callbacks enabled." -ForegroundColor Yellow
        deactivate
        exit
    }
    
    # Create output directory
    if (-not (Test-Path $OutputDir)) {
        Write-Host "Creating output directory: $OutputDir" -ForegroundColor Green
        New-Item -Path $OutputDir -ItemType Directory -Force | Out-Null
    }
      # Generate training visualizations
    Write-Host "`nGenerating training visualizations..." -ForegroundColor Cyan
    python scripts/visualize_advanced_training.py --metrics $BatchMetricsPath --time_logs $TimeLogsPath --lr_logs $LRLogsPath --output $OutputDir
    
    # Create separate directories for different types of visualizations
    $AttentionOutputDir = Join-Path (Split-Path -Parent $OutputDir) "attention_visualizations"
    $AnalysisOutputDir = Join-Path (Split-Path -Parent $OutputDir) "training_time_analysis"
    
    # Create directories if they don't exist
    if (-not (Test-Path $AttentionOutputDir)) {
        New-Item -Path $AttentionOutputDir -ItemType Directory -Force | Out-Null
    }
    if (-not (Test-Path $AnalysisOutputDir)) {
        New-Item -Path $AnalysisOutputDir -ItemType Directory -Force | Out-Null
    }
    
    # Generate attention visualizations if model and features are provided or RunAll is specified
    $generateAttention = $RunAll -or (Test-Path $ModelPath -and Test-Path $StaticFeaturesPath -and Test-Path $DynamicFeaturesPath)
    
    if ($generateAttention) {
        if (Test-Path $ModelPath -and Test-Path $StaticFeaturesPath -and Test-Path $DynamicFeaturesPath) {
            Write-Host "`nGenerating attention visualizations..." -ForegroundColor Cyan
            python scripts/enhanced_attention_visualization.py --model $ModelPath --static_features $StaticFeaturesPath --dynamic_features $DynamicFeaturesPath --output $AttentionOutputDir
            
            if (Test-Path (Join-Path $AttentionOutputDir "attention_dashboard.png")) {
                Write-Host "Attention visualizations generated successfully in $AttentionOutputDir" -ForegroundColor Green
            } else {
                Write-Host "Failed to generate attention visualizations. Check the error logs." -ForegroundColor Red
            }
        } else {
            Write-Host "`nSkipping attention visualization: Model or feature files not found" -ForegroundColor Yellow
            foreach ($path in @($ModelPath, $StaticFeaturesPath, $DynamicFeaturesPath)) {
                if (-not (Test-Path $path)) {
                    Write-Host "  Missing: $path" -ForegroundColor Yellow
                }
            }
        }
    }
      # Generate training time analysis
    Write-Host "`nGenerating training time analysis..." -ForegroundColor Cyan
    python scripts/analyze_training_time.py --time_logs $TimeLogsPath --metrics_logs $BatchMetricsPath --output $AnalysisOutputDir
    
    if (Test-Path (Join-Path $AnalysisOutputDir "training_time_dashboard.png")) {
        Write-Host "Training time analysis generated successfully in $AnalysisOutputDir" -ForegroundColor Green
        
        # Check if there are optimization recommendations
        $recommendationsFile = Join-Path $AnalysisOutputDir "training_speedup_recommendations.json"
        if (Test-Path $recommendationsFile) {
            Write-Host "`nTraining optimization recommendations found!" -ForegroundColor Cyan
            Write-Host "Check '$recommendationsFile' for suggestions to improve training performance." -ForegroundColor Cyan
        }
    } else {
        Write-Host "Failed to generate training time analysis. Check the error logs." -ForegroundColor Red
    }
    
    # Consolidate all results
    Write-Host "`nAll visualizations and analyses have been generated:" -ForegroundColor Green
    Write-Host "  ✓ Basic training visualizations: $OutputDir" -ForegroundColor Green
    Write-Host "  ✓ Training time analysis: $AnalysisOutputDir" -ForegroundColor Green
    
    if ($generateAttention -and (Test-Path (Join-Path $AttentionOutputDir "attention_dashboard.png"))) {
        Write-Host "  ✓ Attention visualizations: $AttentionOutputDir" -ForegroundColor Green
    }
    
    # Ask to open directories
    Write-Host "`nDo you want to open the visualization directories? (y/n)" -ForegroundColor Cyan
    $openDirs = Read-Host
    
    if ($openDirs -eq "y") {
        Start-Process explorer.exe -ArgumentList (Resolve-Path $OutputDir)
        Start-Process explorer.exe -ArgumentList (Resolve-Path $AnalysisOutputDir)
        
        if ($generateAttention -and (Test-Path $AttentionOutputDir)) {
            Start-Process explorer.exe -ArgumentList (Resolve-Path $AttentionOutputDir)
        }
    }
    
    # Deactivate the environment
    deactivate
} else {
    Write-Host "`nVirtual environment not found at .\env\Scripts\activate.ps1" -ForegroundColor Red
    Write-Host "Please create and set up the virtual environment first." -ForegroundColor Yellow
}

Write-Host "`nVisualization process completed." -ForegroundColor Cyan
Write-Host "You can use these visualizations for your research paper or presentations." -ForegroundColor Cyan
