# PowerShell script to clean up redundant Phase 4 documentation files
# after merging them into phase_4_comprehensive.md

Write-Host "Cleaning up redundant Phase 4 documentation files..." -ForegroundColor Cyan

# Define the path to the docs directory
$docsDir = "c:\Users\ujwal\OneDrive\Documents\GitHub\Cross_Attention_CNN_Research_Execution\docs"

# Define the files to be removed (leaving the comprehensive file)
$filesToRemove = @(
    "phase_4.3_training_system_logging.md",
    "phase_4.3_training_system_integration.md",
    "phase_4.3_completion_summary.md",
    "phase_4.3_testing_guide.md"
)

# Check if the comprehensive file exists
if (Test-Path "$docsDir\phase_4_comprehensive.md") {
    # Remove redundant files
    foreach ($file in $filesToRemove) {
        $filePath = "$docsDir\$file"
        if (Test-Path $filePath) {
            Remove-Item -Path $filePath -Force
            Write-Host "Removed $file" -ForegroundColor Green
        } else {
            Write-Host "File not found: $file" -ForegroundColor Yellow
        }
    }
    
    # Keep the main phase_4_training_pipeline.md file as it may be referenced elsewhere
    Write-Host "Note: Kept phase_4_training_pipeline.md as it may be referenced elsewhere" -ForegroundColor Cyan
    Write-Host "To remove it manually, use:" -ForegroundColor Cyan
    Write-Host "Remove-Item -Path `"$docsDir\phase_4_training_pipeline.md`" -Force" -ForegroundColor Gray
    
    Write-Host "`nDocumentation cleanup complete!" -ForegroundColor Green
    Write-Host "The comprehensive documentation is available at:" -ForegroundColor Green
    Write-Host "$docsDir\phase_4_comprehensive.md" -ForegroundColor Green
} else {
    Write-Host "Error: Comprehensive documentation file not found!" -ForegroundColor Red
    Write-Host "Please ensure phase_4_comprehensive.md was created before running this script." -ForegroundColor Red
}
