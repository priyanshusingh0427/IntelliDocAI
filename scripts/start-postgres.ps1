# Starts the portable PostgreSQL 17 server used by IntelliDoc AI (runs as your user, no admin).
# Binaries: %LOCALAPPDATA%\Programs\PostgreSQL17   Data: %LOCALAPPDATA%\IntelliDocAI\pgdata
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\start-postgres.ps1
$bin  = "$env:LOCALAPPDATA\Programs\PostgreSQL17\bin"
$data = "$env:LOCALAPPDATA\IntelliDocAI\pgdata"
$log  = "$env:LOCALAPPDATA\IntelliDocAI\postgres.log"

& "$bin\pg_isready.exe" -h 127.0.0.1 -p 5432 -q
if ($LASTEXITCODE -eq 0) { Write-Host "PostgreSQL is already running on 127.0.0.1:5432"; exit 0 }

# Start detached (hidden window) so the server keeps running after this script ends.
Start-Process -FilePath "$bin\pg_ctl.exe" -ArgumentList @('-D', "`"$data`"", '-l', "`"$log`"", 'start') -WindowStyle Hidden
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Milliseconds 500
    & "$bin\pg_isready.exe" -h 127.0.0.1 -p 5432 -q
    if ($LASTEXITCODE -eq 0) { Write-Host "PostgreSQL started on 127.0.0.1:5432"; exit 0 }
}
Write-Host "PostgreSQL did not start. See $log"
exit 1
