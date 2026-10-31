"""Waits for extra_models stage to finish, then runs pooled-extra, HPT, AUG."""
import subprocess, sys, time
from pathlib import Path

PY = sys.executable
LOG = Path("study_results/tail2.log")

def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    LOG.open("a").write(line + "\n")

def extra_alive():
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
        "Where-Object {$_.CommandLine -like '*extra_models*'} | Measure-Object | "
        "Select-Object -ExpandProperty Count"], capture_output=True, text=True)
    return r.stdout.strip() not in ("0", "")

while extra_alive():
    time.sleep(120)
log("extra stage finished; starting remaining stages")
for stage, cmd in [
    ("extra-pooled", [PY, "pooled.py", "--stage", "extra"]),
    ("hpt", [PY, "hpt.py"]),
    ("aug", [PY, "aug.py"]),
]:
    log(f"{stage} start")
    subprocess.run(cmd)
    log(f"{stage} done")
log("TAIL2 COMPLETE")
