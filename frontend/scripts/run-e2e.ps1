# Orchestration script for Nexora AI end-to-end integration test
$ErrorActionPreference = "Stop"

$repoRoot = "C:\Users\Dell\Nexora-AI"
$frontendDir = Join-Path $repoRoot "frontend"
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$viteJs = Join-Path $frontendDir "node_modules\vite\bin\vite.js"

$env:PYTHONPATH = (Join-Path $repoRoot "src")
$backendOutLog = Join-Path $repoRoot "backend_e2e_out.log"
$backendErrLog = Join-Path $repoRoot "backend_e2e_err.log"
$frontendOutLog = Join-Path $repoRoot "frontend_e2e_out.log"
$frontendErrLog = Join-Path $repoRoot "frontend_e2e_err.log"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Starting Nexora AI Backend & Frontend Preview Servers " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# Clean up any stale processes on ports 8000 or 4173
Get-NetTCPConnection -LocalPort 8000, 4173 -ErrorAction SilentlyContinue | ForEach-Object {
    try { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue } catch {}
}

$backendProc = $null
$frontendProc = $null

try {
    Write-Host "[1/4] Launching FastAPI backend server on http://127.0.0.1:8000..." -ForegroundColor Yellow
    $backendProc = Start-Process -FilePath $pythonExe `
        -ArgumentList "-m", "uvicorn", "nexora.api.app:app", "--host", "127.0.0.1", "--port", "8000" `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $backendOutLog `
        -RedirectStandardError $backendErrLog `
        -PassThru

    Write-Host "[2/4] Launching Vite preview server on http://127.0.0.1:4173..." -ForegroundColor Yellow
    $frontendProc = Start-Process -FilePath "node" `
        -ArgumentList "$viteJs", "preview", "--port", "4173", "--host", "127.0.0.1" `
        -WorkingDirectory $frontendDir `
        -RedirectStandardOutput $frontendOutLog `
        -RedirectStandardError $frontendErrLog `
        -PassThru

    # Wait for both servers to be ready
    Write-Host "[3/4] Polling server health endpoints..." -ForegroundColor Yellow
    $backendReady = $false
    for ($i = 0; $i -lt 30; $i++) {
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 2 -ErrorAction Stop
            if ($resp.status -eq "ok") {
                $backendReady = $true
                break
            }
        } catch {
            Start-Sleep -Milliseconds 400
        }
    }

    if (-not $backendReady) {
        if (Test-Path $backendErrLog) {
            Write-Host "--- Backend Stderr Output ---" -ForegroundColor Red
            Get-Content $backendErrLog | Write-Host -ForegroundColor Red
        }
        if (Test-Path $backendOutLog) {
            Write-Host "--- Backend Stdout Output ---" -ForegroundColor Red
            Get-Content $backendOutLog | Write-Host -ForegroundColor Red
        }
        throw "FastAPI backend failed to start on port 8000 within 12 seconds."
    }
    Write-Host "  -> Backend is healthy on port 8000!" -ForegroundColor Green

    $frontendReady = $false
    for ($i = 0; $i -lt 30; $i++) {
        try {
            $resp = Invoke-WebRequest -Uri "http://127.0.0.1:4173/" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
            if ($resp.StatusCode -eq 200) {
                $frontendReady = $true
                break
            }
        } catch {
            Start-Sleep -Milliseconds 400
        }
    }

    if (-not $frontendReady) {
        throw "Vite preview server failed to respond on port 4173 within 12 seconds."
    }
    Write-Host "  -> Vite preview server is responsive on port 4173!" -ForegroundColor Green

    Write-Host "[4/4] Executing test-e2e.mjs test suite..." -ForegroundColor Yellow
    & node (Join-Path $frontendDir "scripts\test-e2e.mjs")
    $testExitCode = $LASTEXITCODE

    if ($testExitCode -ne 0) {
        throw "test-e2e.mjs exited with failure code: $testExitCode"
    }

    Write-Host "Integration verification completed successfully!" -ForegroundColor Green
    exit 0
}
finally {
    Write-Host "Cleaning up test processes..." -ForegroundColor Gray
    if ($backendProc -and -not $backendProc.HasExited) {
        Stop-Process -Id $backendProc.Id -Force -ErrorAction SilentlyContinue
    }
    if ($frontendProc -and -not $frontendProc.HasExited) {
        Stop-Process -Id $frontendProc.Id -Force -ErrorAction SilentlyContinue
    }
    # Double check no lingering sockets
    Get-NetTCPConnection -LocalPort 8000, 4173 -ErrorAction SilentlyContinue | ForEach-Object {
        try { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue } catch {}
    }
}
