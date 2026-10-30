@echo off
setlocal
set "PYTHONUTF8=1"
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Project Python environment is missing.
    pause
    exit /b 1
)
if not exist "input\sample.mp4" (
    echo Input video is missing: input\sample.mp4
    pause
    exit /b 1
)
set "sample_output=outputs\selected-%RANDOM%-%RANDOM%"
echo Input: input\sample.mp4
echo Output: %sample_output%
echo Small-object mode: select a tight box around each target.
echo Left click = include. Right click = exclude, NOT add another object.
echo Press 1, 2 or 3 to choose the object ID, then select it with the mouse.
echo Previous selections are kept. Enter processes the selected objects.
".venv\Scripts\python.exe" main.py run --video "input\sample.mp4" --output "%sample_output%" --tracks --transparent --small-objects
set "sample_exit=%ERRORLEVEL%"
if not "%sample_exit%"=="0" (
    echo Processing failed. See the error above.
) else (
    echo Completed. Results are in %sample_output%
)
pause
exit /b %sample_exit%
