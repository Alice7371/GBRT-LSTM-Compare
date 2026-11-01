"""Fix short chunks: download each missing tail range to partN.tail, append,
then assemble the full zip and run the extract+eval+close chain."""
import subprocess
import sys
import time
from pathlib import Path

D = Path("data_official")
ZIP = D / "modelOutput_daymet.zip"
TARGET = 4207763546
URL = ("https://zenodo.org/records/15529996/files/"
       "basin_timeseries_v1p2_modelOutput_daymet.zip?download=1")
LOG = Path("study_results/parallel_dl.log")

def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    LOG.open("a", encoding="utf-8").write(line + "\n")

# chunk bounds from parallel_dl.log
CHUNKS = [
    (2298384160, 2616614057),
    (2616614058, 2934843955),
    (2934843956, 3253073853),
    (3253073854, 3571303751),
    (3571303752, 3889533649),
    (3889533650, 4207763545),
]

# 1) fix short chunks
for i, (a, b) in enumerate(CHUNKS):
    part = D / f".part{i}"
    have = part.stat().st_size if part.exists() else 0
    need = b - a + 1
    if have >= need:
        continue
    tail_lo = a + have
    tail = D / f".part{i}.tail"
    log(f"part{i} short by {need - have}; fetching [{tail_lo},{b}]")
    r = subprocess.run(["curl", "-sL", "--retry", "20", "--retry-delay", "5",
                        "--retry-all-errors", "-r", f"{tail_lo}-{b}",
                        "-x", "http://127.0.0.1:7897", URL, "-o", str(tail)],
                       capture_output=True, text=True)
    got = tail.stat().st_size if tail.exists() else 0
    if got != need - have:
        log(f"part{i} tail got {got}, need {need - have} — abort")
        sys.exit(1)
    with part.open("ab") as f:
        f.write(tail.read_bytes())
    tail.unlink()
    log(f"part{i} fixed -> {part.stat().st_size}")

# 2) verify all chunks complete
total = 0
for i, (a, b) in enumerate(CHUNKS):
    part = D / f".part{i}"
    need = b - a + 1
    have = part.stat().st_size if part.exists() else 0
    assert have == need, f"part{i} still short: {have}/{need}"
    total += have
log(f"all chunks complete: {total} bytes")

# 3) assemble
assert ZIP.stat().st_size + total == TARGET, "base+chunks != target"
with ZIP.open("ab") as out:
    for i in range(len(CHUNKS)):
        part = D / f".part{i}"
        out.write(part.read_bytes())
        part.unlink()
final = ZIP.stat().st_size
log(f"assembled zip = {final} (target {TARGET})")
assert final == TARGET

# 4) extract
log("extracting (7z)...")
r = subprocess.run([r"C:\Users\alice\AppData\Local\Microsoft\WindowsApps\7z.exe",
                    "x", "-y", f"-o{ZIP.parent}\\modelOutput_daymet", str(ZIP)],
                   capture_output=True, text=True)
n_files = len(list(ZIP.parent.glob("**/*_modelOutput.txt")))
log(f"7z exit {r.returncode}; {n_files} model-output files")
if n_files < 100:
    sys.exit(1)

# 5) evaluate official models under our protocol
log("evaluating official models...")
subprocess.run([sys.executable, "eval_official.py"])
log("evaluation done")

# 6) close Clash (user request)
log("closing Clash")
subprocess.run(["powershell", "-NoProfile", "-Command",
                "Get-Process | Where-Object {$_.Name -like '*clash*'} | "
                "Stop-Process -Force"], capture_output=True)
log("ALL DONE")
