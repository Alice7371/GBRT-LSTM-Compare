"""Parallel range downloader: splits the remaining zenodo zip range into 6
concurrent chunks (bypasses single-stream throttling), concatenates onto the
base zip, verifies size, extracts, evaluates, closes Clash."""
import subprocess, sys, time
from pathlib import Path

PY = sys.executable
ZIP = Path("data_official/modelOutput_daymet.zip")
TARGET = 4207763546
URL = ("https://zenodo.org/records/15529996/files/"
       "basin_timeseries_v1p2_modelOutput_daymet.zip?download=1")
LOG = Path("study_results/parallel_dl.log")
NCHUNK = 6

def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    LOG.open("a", encoding="utf-8").write(line + "\n")

def running(pattern):
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_Process -Filter \"name='python.exe' or name='curl.exe'\" | "
        f"Where-Object {{$_.CommandLine -like '*{pattern}*'}} | Measure-Object | "
        "Select-Object -ExpandProperty Count"], capture_output=True, text=True)
    return r.stdout.strip() not in ("0", "")

# 0) stop the slow single-stream finisher if alive
for pat in ("finisher.py", "modelOutput_daymet.zip?download=1"):
    while running(pat):
        subprocess.run(["powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Process -Filter \"name='python.exe' or name='curl.exe'\" | "
            "Where-Object {$_.CommandLine -like '*" + pat + "*'} | "
            "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"],
            capture_output=True)
        time.sleep(3)
log(f"single-stream stopped; base zip = {ZIP.stat().st_size}")

# 1) parallel chunk downloads
S = ZIP.stat().st_size
lo, hi = S, TARGET - 1
step = (hi - lo + 1) // NCHUNK + 1
procs = []
for i in range(NCHUNK):
    a = lo + i * step
    b = min(hi, a + step - 1)
    if a > hi:
        break
    part = Path(f"data_official/.part{i}")
    have = part.stat().st_size if part.exists() else 0
    need = b - a + 1
    if have >= need:
        log(f"chunk {i} complete ({need} bytes)")
        continue
    a2 = a + have  # resume per-chunk
    procs.append((i, subprocess.Popen(
        ["curl", "-sL", "--retry", "20", "--retry-delay", "5",
         "--retry-all-errors", "-r", f"{a2}-{b}",
         "-x", "http://127.0.0.1:7897", URL, "-o", str(part)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)))
    log(f"chunk {i}: [{a2},{b}] {need} bytes")

fails = 0
for i, p in procs:
    rc = p.wait()
    if rc != 0:
        fails += 1
    log(f"chunk {i} exit {rc}")
if fails:
    log(f"{fails} chunks failed — rerun this script to resume")
    sys.exit(1)

# 2) assemble
total = sum(Path(f"data_official/.part{i}").stat().st_size
            for i in range(NCHUNK) if Path(f"data_official/.part{i}").exists())
log(f"chunks total {total}; appending to base ({S}) -> {S + total}")
if S + total < TARGET:
    log("size mismatch — abort before assembly")
    sys.exit(1)
with ZIP.open("ab") as out:
    for i in range(NCHUNK):
        part = Path(f"data_official/.part{i}")
        if part.exists():
            out.write(part.read_bytes())
            part.unlink()
final = ZIP.stat().st_size
log(f"assembled zip = {final} (target {TARGET})")
if final != TARGET:
    log("final size mismatch — abort")
    sys.exit(1)

# 3) extract
log("extracting (7z)...")
r = subprocess.run([r"C:\Users\alice\AppData\Local\Microsoft\WindowsApps\7z.exe",
                    "x", "-y", f"-o{ZIP.parent}\\modelOutput_daymet", str(ZIP)],
                   capture_output=True, text=True)
n_files = len(list(ZIP.parent.glob("**/*_modelOutput.txt")))
log(f"7z exit {r.returncode}; {n_files} model-output files")

# 4) evaluate
if n_files > 100:
    log("evaluating official models...")
    subprocess.run([PY, "eval_official.py"])
    log("evaluation done")

# 5) close Clash (user request)
log("closing Clash")
subprocess.run(["powershell", "-NoProfile", "-Command",
                "Get-Process | Where-Object {$_.Name -like '*clash*'} | "
                "Stop-Process -Force"], capture_output=True)
log("PARALLEL DL DONE")
