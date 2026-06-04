@echo off
setlocal
cd /d "%~dp0"
echo Compiling 规划.md and 教程.md ...
python compile_docs.py
if errorlevel 1 exit /b 1
echo Done: 规划.pdf, 教程.pdf
exit /b 0
