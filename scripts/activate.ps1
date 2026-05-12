# Quick activation script — run with: .\scripts\activate.ps1
if (-not (Test-Path .\.venv)) {
    Write-Host "No .venv found. Run 'uv venv' first." -ForegroundColor Yellow
    exit 1
}
.\.venv\Scripts\Activate.ps1
Write-Host "Environment activated. Python: $(python --version)" -ForegroundColor Green