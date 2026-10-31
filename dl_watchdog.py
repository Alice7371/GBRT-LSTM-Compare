"""Waits for the 4GB model-output download, extracts, evaluates official models
under our protocol, then closes Clash (user request)."""
import subprocess, sys, time
from pathlib import Path

ZIP = Path("data_official/modelOutput_daymet.zip")
TARGET = 4013 * 1048576
LOG = Path("study_results/dl_watchdog.log")

def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    LOG.open("a", encoding="utf-8").write(line + "\n")

def running(pattern):
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
        f"Where-Object {{$_.CommandLine -like '*{pattern}*'}} | Measure-Object | "
        "Select-Object -ExpandProperty Count"], capture_output=True, text=True)
    return r.stdout.strip() not in ("0", "")

# 1) wait for download completion — SIZE ONLY (process-based check misfired
#    when curl dropped mid-transfer and closed Clash prematurely; never again)
while True:
    sz = ZIP.stat().st_size if ZIP.exists() else 0
    if sz >= TARGET * 0.99:
        break
    time.sleep(120)
log(f"download complete by size: {sz/1048576:.0f} MB")

# 2) extract with 7z (streamed, no full verification to save time)
log("extracting with 7z...")
r = subprocess.run([r"C:\Users\alice\AppData\Local\Microsoft\WindowsApps\7z.exe",
                    "x", "-y", f"-o{ZIP.parent}\modelOutput_daymet", str(ZIP)],
                   capture_output=True, text=True)
log(f"7z exit {r.returncode}; files: {len(list(ZIP.parent.glob('modelOutput_daymet/**/*.txt')))} ")

# 3) evaluate official models under our protocol
log("evaluating official models...")
subprocess.run([sys.executable, "eval_official.py"])
log("official models evaluated")

# 4) close Clash (user instruction)
log("closing Clash per user request")
subprocess.run(["powershell", "-NoProfile", "-Command",
                "Stop-Process -Name 'Clash Verge' -Force -ErrorAction SilentlyContinue; "
                "Stop-Process -Name 'clash-verge' -Force -ErrorAction SilentlyContinue; "
                "Stop-Process -Name 'clash-meta' -Force -ErrorAction SilentlyContinue; "
                "Stop-Process -Name 'verge-mihomo' -Force -ErrorAction SilentlyContinue"], capture_output=True)
log("DONE — clash closed")
