"""Analysis for the multi-horizon study (upgraded per review).

Reads study_results/{main,extend_*,pooled_*}.csv and produces:
- per-track-per-horizon medians with bootstrap 95% CI
- predictability decay curves (NSE vs horizon, snow/rain strata)
- paired Wilcoxon + sign tests, BH-FDR correction, effect sizes
  (rank-biserial correlation, Cohen's d)
- win rates, FE/pooling gains, pooled vs per-basin comparisons
- flow-regime strata (intermittency, CV) and snow classification
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import data as D

OUTDIR = Path("study_results")
RNG = np.random.default_rng(0)

# per-basin descriptors, cached to disk (loads 669 CSVs once)
def basin_descriptors() -> pd.DataFrame:
    cache = OUTDIR / "basin_desc.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"station": str})
    rows = []
    for p in sorted(Path(D.DATA_DIR).glob("*.csv")):
        s = p.stem
        df = pd.read_csv(p)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        q = df["Discharge"].values
        split = D.split_index(len(q))
        q_tr, t_tr = q[:split], df.iloc[:split]
        tmean = (t_tr["Tmax"] + t_tr["Tmin"]).values / 2.0
        rows.append({
            "station": s,
            "zero_frac": float((q_tr == 0).mean()),
            "cv": q_tr.std() / (q_tr.mean() + 1e-9),
            "frozen_frac": float((tmean < 0).mean()),   # snow proxy
            "log_mean_q": float(np.log10(q_tr.mean() + 1.0)),
        })
    desc = pd.DataFrame(rows)
    desc.to_csv(cache, index=False)
    return desc


def attach(df, desc=None):
    desc = desc if desc is not None else basin_descriptors()
    df = df.copy()
    df["station"] = df["station"].str.zfill(8)
    return df.merge(desc, on="station", how="left")


def snow_class(desc):
    """Snow-dominated if >15% of training days below 0 C, rain otherwise."""
    return desc.assign(snow=desc.frozen_frac > 0.15)


def load_all():
    frames = []
    for pat in ("main.shard*.csv", "extend_*.csv", "pooled_*.csv", "gr4j.csv"):
        frames += [pd.read_csv(f, dtype={"station": str})
                   for f in sorted(OUTDIR.glob(pat))]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["station"] = df["station"].str.zfill(8)
    # concurrent appenders can duplicate (station, track, horizon, seed, years)
    df = df.drop_duplicates(subset=["station", "track", "horizon", "seed", "years"])
    return df


# ---------- statistics ----------

def boot_ci(x, n=1000, alpha=0.05):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return (np.nan, np.nan)
    meds = [np.median(RNG.choice(x, len(x), replace=True)) for _ in range(n)]
    return (np.quantile(meds, alpha / 2), np.quantile(meds, 1 - alpha / 2))


def paired_stats(df, a, b, horizon, metric="nse"):
    piv = df[df.horizon == horizon].pivot_table(
        index="station", columns="track", values=metric, aggfunc="median")
    if a not in piv or b not in piv:
        return None
    diff = (piv[a] - piv[b]).dropna()
    n = len(diff)
    if n < 10:
        return None
    w = stats.wilcoxon(diff, alternative="greater")
    # rank-biserial from the one-sided Wilcoxon statistic
    rb = 2 * w.statistic / (n * (n + 1) / 2) - 1 if n else np.nan
    cohen_d = diff.mean() / (diff.std(ddof=1) + 1e-12)
    ci_lo, ci_hi = boot_ci(diff.values)
    return dict(n=n, wins=int((diff > 0).sum()), median_delta=float(diff.median()),
                ci=(ci_lo, ci_hi), p=w.pvalue, rank_biserial=rb, cohens_d=float(cohen_d))


def fdr_report(rows, alpha=0.05):
    """Benjamini-Hochberg over the collected test rows."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    p = df.p.values
    order = np.argsort(p)
    m = len(p)
    thresh = alpha * (np.arange(1, m + 1) / m)[order]
    passed = np.empty(m, bool)
    sorted_p = p[order]
    running = sorted_p <= thresh
    # classic BH: largest k where p(k) <= k/m*alpha; reject 1..k
    k_max = 0
    for i in range(m):
        if running[i]:
            k_max = i + 1
    sig = order[:k_max] if k_max else []
    df["significant_fdr"] = False
    df.loc[sig, "significant_fdr"] = True
    return df


# ---------- tables ----------

def med_ci_table(df, metric="nse"):
    rows = []
    for (track, h), g in df.groupby(["track", "horizon"]):
        lo, hi = boot_ci(g[metric].values)
        rows.append(dict(track=track, horizon=h,
                         median=float(g[metric].median()),
                         ci_lo=lo, ci_hi=hi, n=len(g)))
    return (pd.DataFrame(rows)
            .sort_values(["track", "horizon"])
            .round(3))


def decay_table(df, metric="nse"):
    piv = df.pivot_table(index="track", columns="horizon", values=metric,
                         aggfunc="median").round(3)
    return piv.reindex(sorted(piv.columns), axis=1)


def pooled_vs_perbasin(df, metric="nse"):
    pairs = [
        ("lstm-pooled-emb", "lstm-raw-zscore"),
        ("lstm-pooled-plain", "lstm-raw-zscore"),
        ("lstm-pooled-emb-fe", "lstm-fe-zscore"),
        ("xgb-pooled-fe", "xgb-fe"),
        ("xgb-pooled-raw", "xgb-raw"),
    ]
    rows = []
    for h in sorted(df.horizon.unique()):
        for a, b in pairs:
            r = paired_stats(df, a, b, h, metric)
            if r:
                rows.append(dict(horizon=h, pooled=a, perbasin=b, **{
                    k: v for k, v in r.items() if k != "ci"},
                    ci_lo=r["ci"][0], ci_hi=r["ci"][1]))
    return fdr_report(rows)


def stratified_decay(df, metric="nse"):
    """Decay curves per snow class."""
    out = {}
    for cls in ("snow", "rain"):
        sub = df[df.snow == (cls == "snow")]
        out[cls] = sub.pivot_table(index="track", columns="horizon",
                                   values=metric, aggfunc="median").round(3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="nse")
    ap.add_argument("--out", default=None, help="write markdown to this file too")
    args = ap.parse_args()
    m = args.metric
    lines = []

    def P(s=""):
        print(s)
        lines.append(str(s))

    df = load_all()
    if df.empty:
        P("no results yet")
        return
    desc = snow_class(basin_descriptors())
    df = attach(df, desc)

    P(f"# Study analysis ({m}) — {len(df)} rows, {df.station.nunique()} basins")
    P(f"\nsnow-dominated basins (frozen_frac>0.15): {int(desc.snow.sum())}/{len(desc)}")

    P("\n## medians with bootstrap 95% CI")
    P(med_ci_table(df, m).to_string(index=False))

    P("\n## predictability decay (median NSE vs horizon)")
    P(decay_table(df, m).to_string())

    P("\n## snow strata decay")
    for cls, tab in stratified_decay(df, m).items():
        P(f"[{cls}]"); P(tab.to_string())

    P("\n## pooled vs per-basin (paired, FDR-controlled)")
    t = pooled_vs_perbasin(df, m)
    if not t.empty:
        P(t.round(4).to_string(index=False))

    P("\n## family & anchor tests (paired, FDR-controlled)")
    rows = []
    tests = [("lstm-raw-zscore", "xgb-raw"), ("lstm-raw-zscore", "lstm-raw-minmax"),
             ("lstm-raw-zscore", "persistence"), ("xgb-fe", "persistence"),
             ("lstm-raw-zscore", "climatology"), ("lstm-raw-zscore", "gr4j"),
             ("lstm-raw-zscore", "mlp-zscore"), ("lstm-raw-zscore", "gru-zscore"),
             ("gru-zscore", "mlp-zscore"), ("lstm-raw-zscore", "attn-zscore"),
             ("mlp-zscore", "persistence"), ("xgb-fe", "climatology"),
             ("lstm-fe-zscore", "lstm-raw-zscore"), ("xgb-fe", "xgb-raw")]
    for h in sorted(df.horizon.unique()):
        for a, b in tests:
            r = paired_stats(df, a, b, h, m)
            if r:
                rows.append(dict(horizon=h, a=a, b=b, **{
                    k: v for k, v in r.items() if k != "ci"},
                    ci_lo=r["ci"][0], ci_hi=r["ci"][1]))
    if rows:
        P(fdr_report(rows).round(4).to_string(index=False))

    P("\n## flow-regime strata (h=1, median NSE)")
    sub = df[df.horizon == 1].copy()
    sub["regime"] = pd.cut(sub.zero_frac, [-1, .02, .2, 1.01],
                           labels=["perennial", "semi-intermittent", "intermittent"])
    P(sub.groupby(["regime", "track"], observed=True)[m].median()
        .unstack("track").round(3).to_string())

    if args.out:
        Path(args.out).write_text("\n".join(lines), encoding="utf-8")
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
