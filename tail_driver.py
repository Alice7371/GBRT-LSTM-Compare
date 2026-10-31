"""Waits for the main queue driver to exit, then:
  1. extend 90/180 single-process slim stride-12 (2-way sharding oversubscribed
     the 8GB card on window-270 LSTMs)
  2. HPT (Optuna 50 trials)
  3. AUG ablation
"""
import subprocess, sys, time

PY = sys.executable

def alive(pattern):
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
        f"Where-Object {{$_.CommandLine -like '*{pattern}*'}} | Measure-Object | "
        "Select-Object -ExpandProperty Count"], capture_output=True, text=True)
    return r.stdout.strip() not in ("0", "")

while alive("queue_driver"):
    time.sleep(180)

stages = [
    ("extend-90-180-slim", [PY, "run_experiments.py", "extend",
                            "--horizons", "90", "180", "--stride", "20",
                            "--slim"]),
    ("extra-perbasin", [PY, "extra_models.py", "--horizons", "1", "7", "30",
                        "--stride", "2", "--models", "mlp", "attn", "gru"]),
    ("extra-pooled", [PY, "pooled.py", "--stage", "extra"]),
    ("hpt", [PY, "hpt.py"]),
    ("aug", [PY, "aug.py"]),
]
for stage, cmd in stages:
    print(f"tail: {stage} start", flush=True)
    subprocess.run(cmd)
    print(f"tail: {stage} done", flush=True)
print("TAIL COMPLETE", flush=True)
