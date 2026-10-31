"""Robust finisher: resume-loop the 4GB download until complete, extract,
evaluate official models, then close Clash. Runs via schtasks (outlives the
ZCode session and its process-pair quirks)."""
import subprocess, sys, time
from pathlib import Path

PY = sys.executable
ZIP = Path("data_official/modelOutput_daymet.zip")
TARGET = 4013 * 1048576
LOG = Path("study_results/finisher.log")

def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    LOG.open("a", encoding="utf-8").write(line + "\n")

def size():
    return ZIP.stat().st_size if ZIP.exists() else 0

def proxy_alive():
    r = subprocess.run(["curl", "-s", "--max-time", "8", "-x", "http://127.0.0.1:7897",
                        "-o", "NUL", "-w", "%{http_code}",
                        "https://zenodo.org/api/records/15529996"],
                       capture_output=True, text=True)
    return r.stdout.strip().startswith("2")

# 1) resume loop until size target reached
while size() < TARGET * 0.99:
    if not proxy_alive():
        log(f"proxy down at {size()/1048576:.0f} MB — waiting for Clash")
        time.sleep(180)
        continue
    log(f"resuming from {size()/1048576:.0f} MB")
    subprocess.run(["curl", "-sL", "-C", "-", "--max-time", "7200",
                    "-x", "http://127.0.0.1:7897",
                    "https://zenodo.org/records/15529996/files/"
                    "basin_timeseries_v1p2_modelOutput_daymet.zip?download=1",
                    "-o", str(ZIP)])
    log(f"curl exited; size now {size()/1048576:.0f} MB")
    time.sleep(10)

log(f"download complete: {size()/1048576:.0f} MB")

# 2) extract
log("extracting (7z)...")
r = subprocess.run([r"C:\Users\alice\AppData\Local\Microsoft\WindowsApps\7z.exe",
                    "x", "-y", f"-o{ZIP.parent}\\modelOutput_daymet", str(ZIP)],
                   capture_output=True, text=True)
n_files = len(list(ZIP.parent.glob("**/*_modelOutput.txt")))
log(f"7z exit {r.returncode}; {n_files} model-output files")

# 3) evaluate under our protocol
if n_files > 100:
    log("evaluating official models...")
    subprocess.run([PY, "eval_official.py"])
    log("evaluation done")

# 4) close Clash (user request)
log("closing Clash")
subprocess.run(["powershell", "-NoProfile", "-Command",
                "Get-Process | Where-Object {$_.Name -like '*clash*'} | "
                "Stop-Process -Force"], capture_output=True)
log("FINISHER DONE")
