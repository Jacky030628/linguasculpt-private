$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (!(Test-Path '.venv\Scripts\python.exe')) { python -m venv .venv }
if (!(Test-Path '.env')) { Copy-Item '.env.example' '.env' }
& '.venv\Scripts\python.exe' -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) { throw 'pip 初始化失败' }
& '.venv\Scripts\python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw '依赖安装失败' }
Write-Host '打开 http://127.0.0.1:8000 。密钥请填写在项目 .env 中。'
& '.venv\Scripts\python.exe' -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
