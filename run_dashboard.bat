@echo off
rem Double-click to open the dashboard in your browser.
cd /d "%~dp0scripts"
python -m streamlit run dashboard.py
pause
