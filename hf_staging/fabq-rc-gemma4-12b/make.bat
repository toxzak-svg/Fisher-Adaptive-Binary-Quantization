@echo off
REM ============================================================
REM FABQ-RC Gemma 4 12B - Windows Makefile equivalent
REM ============================================================
REM
REM Usage (from PowerShell or cmd in this folder):
REM   .\make.bat install
REM   .\make.bat test-cuda
REM   .\make.bat bucket
REM   .\make.bat notebook-text
REM   .\make.bat notebook-stream
REM   .\make.bat push
REM   .\make.bat push-dry-run
REM   .\make.bat clean
REM
REM Equivalent GNU make targets are documented in QUICKSTART.md
REM and the parent Makefile. This .bat file just calls Python
REM directly with the same logic, since 'make' isn't on most
REM Windows machines by default.
REM ============================================================

setlocal enabledelayedexpansion

REM Load .env if it exists
if exist .env (
    for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
        if not "%%a"=="" set %%a=%%b
    )
)

if "%HF_TOKEN%"=="" set "HF_TOKEN=YOUR_TOKEN_HERE"
if "%HF_BUCKET%"=="" set "HF_BUCKET=toxzak/gemma-4-12B-it-fabq-rc-bucket"
if "%SOURCE_MODEL%"=="" set "SOURCE_MODEL=google/gemma-4-12B-it"

set "PYTHON=python"

REM ---- Help ----
if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="/?" goto help

REM ---- Targets ----
if "%1"=="install" goto install
if "%1"=="test-cuda" goto test_cuda
if "%1"=="bucket" goto bucket
if "%1"=="notebook-text" goto notebook_text
if "%1"=="notebook-stream" goto notebook_stream
if "%1"=="push" goto push
if "%1"=="push-dry-run" goto push_dry_run
if "%1"=="clean" goto clean

echo Unknown target: %1
echo Run `.\make.bat help` for the list.
exit /b 1

:help
echo.
echo FABQ-RC Gemma 4 12B - available targets:
echo.
echo   install            Install Python dependencies
echo   test-cuda          Build CUDA extension and run tests
echo   bucket             One-time: populate the HF bucket
echo   notebook-text      Launch the text-only quantization notebook
echo   notebook-stream    Launch the streaming + native-quantized notebook
echo   push               Re-push both HF repos (curated)
echo   push-dry-run       Show what would be uploaded
echo   clean              Remove staging dir + extension build artifacts
echo.
echo Required env vars:
echo   HF_TOKEN     HuggingFace token (gated model access)
echo   CUDA_HOME    CUDA toolkit path (only for test-cuda / bucket)
echo.
echo Set them in .env (copy from .env.example) or as Windows env vars.
exit /b 0

REM ---- Setup ----
:install
echo Installing Python dependencies...
%PYTHON% -m pip install --upgrade pip
%PYTHON% -m pip install -r requirements.txt
if errorlevel 1 goto fail
echo.
echo Install complete. Next: set HF_TOKEN, then run make.bat test-cuda
exit /b 0

REM ---- CUDA extension ----
:test_cuda
if "%HF_TOKEN%"=="YOUR_TOKEN_HERE" (
    echo WARNING: HF_TOKEN not set. Continuing anyway - the test does not need it.
)
echo.
echo Building fabq_rc_cuda extension (first run, 2-5 min)...
cd streaming\fabq_rc_cuda
%PYTHON% setup.py build_ext --inplace
if errorlevel 1 ( cd ..\.. & goto fail )
cd ..\..
echo.
echo Running numerical tests...
cd streaming\fabq_rc_cuda
%PYTHON% tests\test_kernel.py
if errorlevel 1 ( cd ..\.. & goto fail )
cd ..\..
echo.
echo CUDA extension built and verified.
exit /b 0

REM ---- Bucket ----
:bucket
if "%HF_TOKEN%"=="YOUR_TOKEN_HERE" (
    echo ERROR: HF_TOKEN not set. Edit .env or set it as an env var.
    exit /b 1
)
echo.
echo Populating HF bucket (one-time, ~30-45 min on A100)...
cd streaming
%PYTHON% build_bucket.py --source %SOURCE_MODEL% --push
if errorlevel 1 ( cd .. & goto fail )
cd ..
echo.
echo Bucket populated: https://huggingface.co/%HF_BUCKET%
exit /b 0

REM ---- Notebooks ----
:notebook_text
echo Launching FABQ-RC-Gemma4-12B.ipynb...
echo   Hardware: needs A100 80GB.
echo   Runtime:  ~30-60 min.
%PYTHON% -m jupyter notebook FABQ-RC-Gemma4-12B.ipynb
exit /b 0

:notebook_stream
echo Launching FABQ-RC-Gemma4-12B-Streaming.ipynb...
echo   Hardware: needs A100 80GB + CUDA toolchain.
echo   Prereq:   bucket must be populated.
echo   Runtime:  ~5-10 min (mostly the in-cell CUDA build on first run).
cd streaming
%PYTHON% -m jupyter notebook FABQ-RC-Gemma4-12B-Streaming.ipynb
exit /b 0

REM ---- Push ----
:push
echo Curating and pushing to HF...
cd ..
%PYTHON% push_to_hf.py
cd gemma4-12b
exit /b 0

:push_dry_run
echo DRY RUN - staging only, no upload...
cd ..
%PYTHON% push_to_hf.py --dry-run
cd gemma4-12b
exit /b 0

REM ---- Clean ----
:clean
echo Cleaning staging dir + extension build artifacts...
if exist hf_staging rmdir /s /q hf_staging
if exist streaming\fabq_rc_cuda\build rmdir /s /q streaming\fabq_rc_cuda\build
del /q streaming\fabq_rc_cuda\_C*.so 2>nul
echo Cleaned.
exit /b 0

:fail
echo.
echo FAILED. Scroll up for the error.
exit /b 1
