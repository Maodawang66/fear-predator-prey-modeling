@echo off
setlocal
cd /d "%~dp0"

echo [1/2] Compiling report.tex with XeLaTeX (latexmk)...
latexmk -xelatex -f -interaction=nonstopmode -file-line-error -synctex=1 report.tex
set ERR=%ERRORLEVEL%

if exist report.pdf (
  echo.
  echo [2/2] Done: report.pdf
  for %%A in (report.pdf) do echo       Size: %%~zA bytes
  exit /b 0
)

echo.
echo Compilation failed. See report.log for details.
exit /b %ERR%
