$ErrorActionPreference = "Stop"

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example"
}

python -m uvicorn app.main:app --host 0.0.0.0 --port 9000 --reload
