@echo off
echo Activating conda environment 'mine2'...
call conda activate mine2

echo Starting GPU test...
python gpu_test.py

pause
