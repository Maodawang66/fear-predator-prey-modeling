@echo off
cd /d "%~dp0"
echo === Deep data analysis (tier 1 + 2) ===
python data\deep_data_analysis.py %*
if errorlevel 1 exit /b 1
echo === Done ===
