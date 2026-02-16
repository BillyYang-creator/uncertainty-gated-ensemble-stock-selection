@echo off
setlocal

cd /d %~dp0

if not exist .venv (
  python -m venv .venv
)

call .venv\Scripts\activate
pip install -r requirements.txt
python main.py inspect-models --root .

echo.
echo If you see factor_count and prediction_shape above, local check passed.
endlocal
