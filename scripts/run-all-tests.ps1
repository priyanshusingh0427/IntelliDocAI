<#
Runs every IntelliDoc AI check and prints a summary. From the project root:

    powershell -ExecutionPolicy Bypass -File scripts\run-all-tests.ps1

Options:
    -SkipE2E      skip the browser end-to-end tests (fastest)
    -WithModels   also run the real-model backend tests (embedding, cross-encoder, Qwen)
    -WithOllama   run the browser tests against the local Ollama model instead of "extractive"

Nothing here touches your real data: backend tests use in-memory SQLite or the separate
intellidoc_test PostgreSQL database; browser tests use a throwaway stack (backend\.e2e-run).
#>
param([switch]$SkipE2E, [switch]$WithModels, [switch]$WithOllama)

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root 'backend'
$frontend = Join-Path $root 'frontend'
$python = Join-Path $root '.venv\Scripts\python.exe'
$results = [System.Collections.Generic.List[object]]::new()

function Invoke-Step([string]$name, [string]$dir, [scriptblock]$command) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    $watch = [Diagnostics.Stopwatch]::StartNew()
    Push-Location $dir
    try { & $command; $code = $LASTEXITCODE } catch { Write-Host $_ -ForegroundColor Red; $code = 1 } finally { Pop-Location }
    $status = if ($code -eq 0) { 'PASS' } else { 'FAIL' }
    $results.Add([pscustomobject]@{ Step = $name; Result = $status; Seconds = [math]::Round($watch.Elapsed.TotalSeconds, 1) })
}

function Add-Skip([string]$name, [string]$reason) {
    Write-Host "`n=== $name === skipped: $reason" -ForegroundColor Yellow
    $results.Add([pscustomobject]@{ Step = $name; Result = "SKIP ($reason)"; Seconds = 0 })
}

$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'

Invoke-Step 'Backend lint (ruff)' $backend { & $python -m ruff check app tests scripts }
Invoke-Step 'Backend tests (SQLite) + coverage' $backend { & $python -m pytest -q --cov --cov-report=term:skip-covered }

# PostgreSQL run: only if the portable server is up and the app password file exists.
$pgReady = Join-Path $env:LOCALAPPDATA 'Programs\PostgreSQL17\bin\pg_isready.exe'
$pwFile = Join-Path $env:LOCALAPPDATA 'IntelliDocAI\app-db-password.txt'
$pgUp = $false
if ((Test-Path $pgReady) -and (Test-Path $pwFile)) {
    & $pgReady -h 127.0.0.1 -p 5432 -q
    $pgUp = ($LASTEXITCODE -eq 0)
}
if ($pgUp) {
    $pw = (Get-Content $pwFile -Raw).Trim()
    Invoke-Step 'Backend tests (PostgreSQL)' $backend {
        $env:TEST_DATABASE_URL = "postgresql+psycopg://intellidoc:$pw@127.0.0.1:5432/intellidoc_test"
        try { & $python -m pytest -q } finally { Remove-Item Env:TEST_DATABASE_URL }
    }
} else {
    Add-Skip 'Backend tests (PostgreSQL)' 'PostgreSQL not running (scripts\start-postgres.ps1)'
}

if ($WithModels) {
    Invoke-Step 'Backend real-model tests' $backend {
        $env:RUN_MODEL_TESTS = '1'
        try { & $python -m pytest -q -k 'real or model' } finally { Remove-Item Env:RUN_MODEL_TESTS }
    }
}

Invoke-Step 'Frontend unit tests (Vitest)' $frontend { npx vitest run }
Invoke-Step 'Frontend production build' $frontend { npx vite build }

if ($SkipE2E) {
    Add-Skip 'Browser end-to-end tests' '-SkipE2E'
} else {
    $label = if ($WithOllama) { 'Browser end-to-end tests (Ollama)' } else { 'Browser end-to-end tests' }
    Invoke-Step $label $frontend {
        if ($WithOllama) { $env:E2E_LLM_PROVIDER = 'ollama' }
        try { npx playwright test --reporter=list } finally { Remove-Item Env:E2E_LLM_PROVIDER -ErrorAction SilentlyContinue }
    }
}

Write-Host "`n=== Summary ===" -ForegroundColor Cyan
$results | Format-Table -AutoSize | Out-String | Write-Host
$failed = @($results | Where-Object { $_.Result -eq 'FAIL' })
if ($failed.Count) {
    Write-Host "$($failed.Count) step(s) FAILED" -ForegroundColor Red
    exit 1
}
Write-Host 'All checks passed' -ForegroundColor Green
exit 0
