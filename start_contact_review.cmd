@echo off
cd /d "%~dp0"
set "PYTHONPATH=%~dp0volley_env_311\Lib\site-packages"
set "PYTHONUTF8=1"
echo Open http://127.0.0.1:5062 and keep this window open.
"%~dp0.python\cpython-3.11.14-windows-x86_64-none\python.exe" -m action_recognition.annotator.contact_spatial_annotator
pause
