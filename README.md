# FSOC Command Center — PySide6

Modern desktop UI replacement for the Tkinter dashboard. The existing OpenCV/NumPy simulation, target motion, disturbance/noise engine, spiral search, beacon detection, Kalman tracking, PID control, BP-2 MP4 loading, and CSV/JSON benchmark logger are retained.

## Run

```bat
python -m pip install -r requirements.txt
python app.py
```

Or double-click `run.bat`.

## Build EXE

Run `build_exe.bat`. The executable is created under `dist\\FSOC_Command_Center\\`.

Tkinter is not imported by the new application.
