"""Stress tests for the structural pair — is it just a sector / factor / few-names bet?"""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config as C
from . import data as D
from .strategy import basket_returns, hedged_spread, ERA


def factor_regression(spread):
    etf = pd.read_csv(C.RAW / "factor_etfs.csv", index_col=0, parse_dates=True).pct_change(fill_method=None)
    X = pd.DataFrame({
        "MKT": etf.SPY,
        "IND_minus_TECH": etf.XLI - etf.XLK,     # is it just industrials vs tech?
        "SIZE": etf.IWM - etf.SPY,
        "MOM": etf.MTUM - etf.SPY,
        "VALUE": etf.IWD - etf.IWF,
        "QUALITY": etf.QUAL - etf.SPY,
    })
    out = {}
    for label, sl in [("pre", slice(None, ERA - pd.Timedelta(days=1))), ("era", slice(ERA, None))]:
        df = pd.concat([spread.rename("y"), X], axis=1).dropna().loc[sl]
        m = sm.OLS(df.y, sm.add_constant(df.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        out[label] = {"alpha_ann_pct": round(100 * 252 * m.params["const"], 1),
                      "alpha_t": round(m.tvalues["const"], 2), "r2": round(m.rsquared, 3),
                      "loadings": {k: round(v, 2) for k, v in m.params.drop("const").items()},
                      "t": {k: round(v, 1) for k, v in m.tvalues.drop("const").items()}}
    return out


def leave_out(px, drop):
    fl0, kn0 = C.FRONTLINE_LONG.copy(), C.KNOWLEDGE_SHORT.copy()
    try:
        C.FRONTLINE_LONG = {k: v for k, v in fl0.items() if k not in drop}
        C.KNOWLEDGE_SHORT = {k: v for k, v in kn0.items() if k not in drop}
        b, _ = basket_returns(px)
        s, _, _ = hedged_spread(b)
    finally:
        C.FRONTLINE_LONG, C.KNOWLEDGE_SHORT = fl0, kn0
    e = s.loc[ERA:]
    return {"dropped": drop, "era_sharpe": round(e.mean() / e.std() * np.sqrt(252), 2),
            "era_ann_ret_pct": round(100 * 252 * e.mean(), 1)}


def random_subbaskets(px, n=500, seed=11):
    rng = np.random.default_rng(seed)
    r = px.pct_change(fill_method=None).clip(-0.5, 1.0).loc["2016":]
    fl = [t for t in C.FRONTLINE_LONG if t in r]
    kn = [t for t in C.KNOWLEDGE_SHORT if t in r]
    spy = r[C.BENCH]
    sh = []
    for _ in range(n):
        f = rng.choice(fl, len(fl) // 2, replace=False)
        k = rng.choice(kn, len(kn) // 2, replace=False)
        b = pd.DataFrame({"FL": r[f].mean(axis=1), "KN": r[k].mean(axis=1), "SPY": spy}).dropna()
        s, _, _ = hedged_spread(b)
        e = s.loc[ERA:]
        sh.append(e.mean() / e.std() * np.sqrt(252))
    sh = np.array(sh)
    return {"n": n, "median_sharpe": round(float(np.median(sh)), 2), "p5_sharpe": round(float(np.percentile(sh, 5)), 2),
            "p95_sharpe": round(float(np.percentile(sh, 95)), 2), "share_positive": round(float((sh > 0).mean()), 3)}


def yearly(spread):
    y = spread.groupby(spread.index.year).apply(lambda x: (1 + x).prod() - 1)
    return {int(k): round(100 * v, 1) for k, v in y.items()}


def run():
    px = D.prices()
    b, _ = basket_returns(px)
    spread, _, _ = hedged_spread(b)
    res = {
        "factor_regression": factor_regression(spread),
        "leave_out": [
            leave_out(px, ["FIX", "EME"]),                          # AI-capex / data-centre trades winners
            leave_out(px, ["FIX", "EME", "LINC", "UTI", "GWW"]),    # top-5 long winners
            leave_out(px, ["CHGG", "TTEC", "FVRR", "GLOB", "COUR"]),  # top-5 short winners
            leave_out(px, ["FIX", "EME", "LINC", "UTI", "GWW", "CHGG", "TTEC", "FVRR", "GLOB", "COUR"]),
        ],
        "random_half_baskets": random_subbaskets(px),
        "spread_by_year_pct": yearly(spread),
    }
    (C.OUT / "robustness.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
