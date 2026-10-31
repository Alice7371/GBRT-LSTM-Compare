"""GR4J lumped conceptual model (Perrin et al. 2003) + Oudin PET + DE calibration.

Per-basin calibration on the train segment only (objective: NSE after a 1-year
warm-up); evaluation on the held-out test segment in h=1 simulation mode.
Latitude for Oudin PET is approximated from the HUC2 prefix of the gauge ID
(the attributes file is not part of our data package; PET seasonal shape
dominates). A basin-area scale factor converts simulated mm/day to ft3/s.

Usage: python gr4j.py [--stations 60] [--workers 8]
Output: study_results/gr4j.csv (one row per station, same schema as pooled).
"""
import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

import data as D

_TOPO = None

def _area_of(station):
    """Real drainage area (km2) from the CAMELS attributes (camels_topo.txt)."""
    global _TOPO
    if _TOPO is None:
        _TOPO = pd.read_csv("data_official/camels_topo.txt", sep=";",
                            dtype={"gauge_id": str})
        _TOPO["gauge_id"] = _TOPO["gauge_id"].str.zfill(8)
    row = _TOPO.loc[_TOPO.gauge_id == station, "area_gages2"]
    return float(row.iloc[0]) if len(row) else 100.0

HUC2_LAT = {1: 45.0, 2: 41.0, 3: 34.0, 5: 40.0, 6: 38.0, 7: 40.0, 8: 32.0,
            9: 39.0, 10: 44.0, 11: 41.0, 12: 37.0, 13: 37.0, 14: 39.0, 15: 33.0}


def oudin_pet(tmean_c, doy, lat_deg):
    """Oudin (2005) PET (mm/day)."""
    phi = np.deg2rad(lat_deg)
    doy = np.asarray(doy, dtype=float)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.25)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365.25 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    re = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(delta) +
        np.cos(phi) * np.cos(delta) * np.sin(ws))
    pet = np.where(re > 0, re / 2.45 * (tmean_c + 5.0) / 100.0, 0.0)
    return np.clip(pet, 0, None)


def _unit_hydrograph(x4):
    """GR4J UH1 (x4 ordinates, 90%) and UH2 (2*x4 ordinates, 10%)."""
    x4i = max(int(round(x4)), 1)
    s = 0.0
    sh1 = []
    for i in range(x4i):
        t = i + 1
        o = np.exp(-np.sqrt(3.0) * (t - 0.5 * x4i + 0.5) / (0.75 * x4i))
        sh1.append(o)
    sh1 = np.array(sh1); sh1 /= sh1.sum()
    sh2 = []
    for i in range(2 * x4i):
        t = i + 1
        if t <= 0.5 * x4i:
            o = 0.0
        else:
            o = np.exp(-np.sqrt(3.0) * (t - 0.5 * x4i - 0.3) / (0.75 * x4i)) * \
                5 / (2 * np.pi)  # 5/2pi scaling as in the reference code
        sh2.append(o)
    sh2 = np.array(sh2); sh2 /= sh2.sum() if sh2.sum() > 0 else 1.0
    return sh1, sh2


def gr4j_run(p_mm, e_mm, x1, x2, x3, x4):
    """GR4J daily model: precipitation/runoff partition, exchange, routing.

    Returns simulated discharge in mm/day (area-normalized).
    """
    n = len(p_mm)
    sh1, sh2 = _unit_hydrograph(x4)

    s = 0.6 * x1           # production store
    pr_series = np.zeros(n)   # total effective rainfall (after exchange)
    for t in range(n):
        p, e = p_mm[t], e_mm[t]
        if p >= e:
            pn = p - e
            u = s / x1
            ps = x1 * (1 - u * u) * np.tanh(pn / x1) / (1 + u * np.tanh(pn / x1))
            s += ps
            pr = pn - ps
        else:
            en = e - p
            es = s * (2.0 - s / x1) * np.tanh(en / x1) / \
                (x1 * (1.0 + (1.0 - s / x1) * np.tanh(en / x1)))
            es = min(es, s)
            s -= es
            pr = 0.0
        f = x2 * (s / x1) ** 3.5   # groundwater exchange
        pr_series[t] = max(pr + f, 0.0)

    # convolve: 90% through UH1 (direct perc), 10% through UH2 (routing store)
    uh1 = np.convolve(pr_series * 0.9, sh1)[:n]
    uh2 = np.convolve(pr_series * 0.1, sh2)[:n]

    q = np.empty(n)
    r = 0.7 * x3           # routing store
    for t in range(n):
        r += uh2[t]
        f2 = x2 * (r / x3) ** 3.5
        r = max(r + f2, 1e-9)
        qr = r * (1.0 - (1.0 + (r / x3) ** 4) ** -0.25)
        r = max(r - qr, 0.0)
        q[t] = uh1[t] + qr
    return np.maximum(q, 0.0)


def calibrate_station(station):
    df = D.load_station(station)
    lat = HUC2_LAT.get(int(station[0:2]), 40.0)
    tmean = (df["Tmax"] + df["Tmin"]).values / 2.0
    pet = oudin_pet(tmean, df["Date"].dt.dayofyear.values, lat)
    prcp_mm = df["Prcp"].values.astype(np.float64)  # CAMELS Daymet forcing: mm/day
    q_cfs = df["Discharge"].values.astype(np.float64)
    split = D.split_index(len(df))

    area_km2 = _area_of(station)

    def neg_nse(theta):
        x1, x2, x3, x4 = 10 ** theta[0], theta[1], 10 ** theta[2], theta[3]
        sim_mm = gr4j_run(prcp_mm, pet, x1, x2, x3, x4)
        sim_cfs = sim_mm * area_km2 / 2.4466  # mm/day over area -> ft3/s
        return -D.nse(q_cfs[365:split], sim_cfs[365:split])

    bounds = [(1.5, 3.9), (-5.0, 3.0), (1.0, 2.9), (0.5, 10.0)]
    res = differential_evolution(neg_nse, bounds, maxiter=60, popsize=12,
                                 seed=0, polish=True)
    x1, x2, x3, x4 = 10 ** res.x[0], res.x[1], 10 ** res.x[2], res.x[3]
    sim_cfs = gr4j_run(prcp_mm, pet, x1, x2, x3, x4) * area_km2 / 2.4466
    return dict(station=station, track="gr4j", horizon=1, seed=0, years=0,
                **D.evaluate(q_cfs[split:], sim_cfs[split:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path("study_results/gr4j.csv")

    ids = sorted(p.stem for p in Path(D.DATA_DIR).glob("*.csv"))
    ids = [s for s in ids if s not in D.EXCLUDE]
    step = max(1, len(ids) // args.stations)
    ids = ids[::step][:args.stations]
    if out.exists():
        done = set(pd.read_csv(out, dtype={"station": str}).station)
        ids = [s for s in ids if s not in done]
    if not ids:
        print("gr4j: all done")
        return
    t0 = time.time()
    with Pool(args.workers) as pool:
        rows = pool.map(calibrate_station, ids, chunksize=1)
    pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
    med = np.median([r["nse"] for r in rows])
    print(f"gr4j: {len(rows)} basins calibrated, median test NSE {med:.3f} "
          f"({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
