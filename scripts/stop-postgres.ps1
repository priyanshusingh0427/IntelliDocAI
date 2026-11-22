# Stops the portable PostgreSQL server cleanly (no data is lost).
$bin  = "$env:LOCALAPPDATA\Programs\PostgreSQL17\bin"
$data = "$env:LOCALAPPDATA\IntelliDocAI\pgdata"
& "$bin\pg_ctl.exe" -D $data -m fast stop
