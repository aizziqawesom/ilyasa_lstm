#!/bin/bash
# Mac version of run_dashboard.bat. Double-click in Finder (see RUN_GUIDE_SIMPLE.md, "Running on a Mac").
# First run: makes a Python virtual environment in .venv and installs the packages (takes a few minutes).
cd "$(dirname "$0")" || exit 1

PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3 not found. Install Python 3.11 or 3.12 from python.org, then run this again."
  read -r -p "Press Enter to close." _; exit 1
fi

if [ ! -d .venv ]; then
  echo "First run: creating .venv with $PY and installing packages..."
  "$PY" -m venv .venv || { read -r -p "venv failed. Press Enter to close." _; exit 1; }
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt || { read -r -p "pip install failed. Press Enter to close." _; exit 1; }
fi

cd scripts || exit 1
echo "Opening dashboard at http://localhost:8501 (close this window or press Ctrl+C to stop)"
../.venv/bin/python -m streamlit run dashboard.py
