"""Part 2 — Turning the labour shift into trades.

Sleeve A  Structural pair: long Frontline Enablers / short Knowledge-Labour Sellers,
          each leg beta-hedged to SPY with lagged rolling betas, vol-targeted.
Sleeve B  Macro timing: Frontline Shift Index (FSI) built only from data that was
          public at the time (publication lags applied) scales the pair 0x..2x.
Sleeve C  Event-driven: frontier-AI release event study on the pair; trade the
          post-event drift (enter at the close of day 0, never before).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from . import data as D

FL_COLOR, KN_COLOR, GREY, INK = "#1f7a4d", "#b3412c", "#777777", "#222222"
ERA = pd.Timestamp(C.AI_ERA_START)
TC_BPS = 10          # one-way cost per unit of gross notional traded
VOL_TARGET = 0.10    # annualised
START = "2016-01-01"


# --------------------------------------------------------------------------- returns
def basket_returns(px):
    r = px.pct_change(fill_method=None).clip(-0.5, 1.0)
    fl = [t for t in C.FRONTLINE_LONG if t in r]
    kn = [t for t in C.KNOWLEDGE_SHORT if t in r]
    out = pd.DataFrame({"FL": r[fl].mean(axis=1), "KN": r[kn].mean(axis=1), "SPY": r[C.BENCH]})
    return out.dropna().loc[START:], r[fl + kn]


def hedged_spread(b, win=252):
    """Each leg hedged to zero SPY beta with betas estimated on data up to t-1."""
    def beta(x):
        cov = x.rolling(win, min_periods=126).cov(b.SPY)
        return (cov / b.SPY.rolling(win, min_periods=126).var()).shift(1)
    bf, bk = beta(b.FL), beta(b.KN)
    spread = (b.FL - bf * b.SPY) - (b.KN - bk * b.SPY)
    return spread.dropna(), bf, bk


def vol_scale(x, win=63):
    vol = x.rolling(win, min_periods=21).std().shift(1) * np.sqrt(252)
    return (VOL_TARGET / vol).clip(0, 3)


# --------------------------------------------------------------------------- signal
def frontline_shift_index(idx):
    """FSI: z-scored blend of three public labour signals, as-of each trading day.

    * Indeed postings spread (frontline/knowledge composite), 13-week change,
      lagged 10 calendar days for publication.
    * CES frontline employment share, 6-month change, lagged 40 days
      (reference month m is published on the first Friday of m+1, ~35d).
    * Young-grad unemployment penalty, 6-month change (sign: a narrowing
      degree advantage = pro-frontline), lagged 40 days.
    Each component is z-scored on an expanding window (no look-ahead).
    """
    ind = D.indeed()
    fl = np.exp(np.log(ind[list(C.INDEED_FRONTLINE)]).mean(axis=1))
    kn = np.exp(np.log(ind[list(C.INDEED_KNOWLEDGE)]).mean(axis=1))
    s1 = np.log(fl / kn).diff(91 // 1).rename("indeed")          # daily series: 91 obs ≈ 13 weeks
    s1.index = s1.index + pd.Timedelta(days=10)

    f = D.fred(["USCONS", "CES6562000001", "CES4300000001", "USLAH", "MANEMP",
                "USINFO", "CES6054000001", "CES5552000001", "CES6055000001",
                "CGBD2024", "LNS14000036"])
    flj = f[["USCONS", "CES6562000001", "CES4300000001", "USLAH", "MANEMP"]].sum(axis=1)
    knj = f[["USINFO", "CES6054000001", "CES5552000001", "CES6055000001"]].sum(axis=1)
    share = (flj / (flj + knj)).dropna()
    s2 = share.diff(6).rename("ces")
    gap = (f.CGBD2024 - f.LNS14000036).dropna().rolling(3).mean()
    s3 = gap.diff(6).rename("grad")
    for s in (s2, s3):
        s.index = s.index + pd.DateOffset(months=1) + pd.Timedelta(days=9)   # ~ release date

    def asof_z(s):
        s = s.dropna()
        z = (s - s.expanding(24).mean()) / s.expanding(24).std()
        return z.reindex(idx.union(z.index)).ffill().reindex(idx)

    comps = pd.concat([asof_z(s1), asof_z(s2), asof_z(s3)], axis=1)
    comps.columns = ["indeed", "ces", "grad"]
    fsi = comps.mean(axis=1, skipna=True)
    return fsi.rename("FSI"), comps


# --------------------------------------------------------------------------- backtest
def kill_switch(fsi):
    """Thesis health check, not a timer: halve risk only after the labour data has
    pointed AWAY from frontline (FSI < -1) for ~2 consecutive months."""
    bad = (fsi.fillna(0) < -1).astype(float).rolling(42, min_periods=42).min().fillna(0)
    return 1 - 0.5 * bad


def production_position(vs, fsi, ev_pos):
    """Core 1x vol-targeted pair, x kill-switch, +0.5x for 10 days after each AI release."""
    return vs * (kill_switch(fsi) + 0.5 * ev_pos)


def perf(r, name):
    r = r.dropna()
    if len(r) < 30:
        return {"name": name}
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    dd = (eq / eq.cummax() - 1).min()
    return {"name": name, "cagr_pct": round(100 * (eq.iloc[-1] ** (1 / yrs) - 1), 2),
            "vol_pct": round(100 * r.std() * np.sqrt(252), 2),
            "sharpe": round(r.mean() / r.std() * np.sqrt(252), 2),
            "max_dd_pct": round(100 * dd, 1), "hit_rate_pct": round(100 * (r > 0).mean(), 1),
            "start": str(r.index[0].date()), "end": str(r.index[-1].date())}


def run_book(spread, pos, name):
    """pos is decided at close t-1 and earns spread on t; costs on changes in gross."""
    pos = pos.reindex(spread.index).ffill().fillna(0)
    gross = 2 * pos.abs()                                   # two legs
    cost = gross.diff().abs().fillna(gross) * TC_BPS / 1e4
    # monthly basket rebalancing cost (≈20% one-way turnover per leg per month)
    month_start = spread.index.to_series().dt.month.diff().fillna(0) != 0
    cost = cost + month_start.astype(float) * gross * 0.20 * TC_BPS / 1e4
    ret = pos.shift(1).fillna(0) * spread - cost
    return ret.rename(name)


def event_study(spread, b, events, pre=5, post=20):
    """Cumulative hedged-spread return around each AI event (day 0 = first trading day ≥ date)."""
    rows, paths = [], {}
    idx = spread.index
    for d, name in events.items():
        d = pd.Timestamp(d)
        pos = idx.searchsorted(d)
        if pos - pre < 0 or pos + post >= len(idx):
            continue
        w = spread.iloc[pos - pre: pos + post + 1]
        w.index = range(-pre, post + 1)
        paths[f"{d.date()} {name}"] = w.cumsum() - w.loc[:-1].sum()
        kn_abn = (b.KN - b.SPY).iloc[pos - 1: pos + 2].sum()
        rows.append({"date": str(d.date()), "event": name,
                     "car_m1_p1_pct": round(100 * w.loc[-1:1].sum(), 2),
                     "car_p1_p10_pct": round(100 * w.loc[1:10].sum(), 2),
                     "car_p1_p20_pct": round(100 * w.loc[1:20].sum(), 2),
                     "knowledge_minus_spy_m1_p1_pct": round(100 * kn_abn, 2)})
    tab = pd.DataFrame(rows)
    P = pd.DataFrame(paths)
    return tab, P


def placebo(spread, n_events, window=(1, 10), n=5000, seed=7, start=ERA):
    """Distribution of the average CAR over random event dates in the same era."""
    rng = np.random.default_rng(seed)
    s = spread.loc[start:]
    c = s.rolling(window[1] - window[0] + 1).sum().shift(-window[1])
    c = c.dropna().values
    draws = rng.choice(c, size=(n, n_events), replace=True).mean(axis=1)
    return draws


def run():
    C.FIG.mkdir(parents=True, exist_ok=True)
    px = D.prices()
    b, single = basket_returns(px)
    spread, bf, bk = hedged_spread(b)
    b = b.loc[spread.index]

    fsi, comps = frontline_shift_index(spread.index)
    vs = vol_scale(spread)

    static = run_book(spread, vs, "A_structural")
    tilt = run_book(spread, vs * (1 + fsi.fillna(0)).clip(0, 2), "B_fsi_tilt")
    timing = run_book(spread, vs * np.sign(fsi.fillna(0)), "B_fsi_timing")

    ev_tab, ev_paths = event_study(spread, b, C.AI_EVENTS)
    # Sleeve C: after each event, hold 1x vol-scaled pair for days +1..+10
    ev_pos = pd.Series(0.0, index=spread.index)
    for d in C.AI_EVENTS:
        p = spread.index.searchsorted(pd.Timestamp(d))
        ev_pos.iloc[p: p + 10] = 1.0          # set at close day 0 -> earns days 1..10
    event_book = run_book(spread, vs * ev_pos, "C_ai_event")
    combo = run_book(spread, production_position(vs, fsi, ev_pos), "PRODUCTION")

    books = pd.concat([static, tilt, timing, event_book, combo], axis=1)
    table = []
    for col in books:
        table.append({**perf(books[col].loc[:ERA - pd.Timedelta(days=1)], col), "period": "pre-AI 2017-2022"})
        table.append({**perf(books[col].loc[ERA:], col), "period": "AI era 2022-12+"})
        table.append({**perf(books[col], col), "period": "full"})
    perf_tab = pd.DataFrame(table)

    # FSI predictive check: IC of FSI vs forward 63d spread return (non-overlapping, monthly)
    fwd = spread[::-1].rolling(63).sum()[::-1].shift(-1)
    mon = pd.concat([fsi, fwd.rename("fwd")], axis=1).dropna().resample("ME").last()
    ic_full = mon.corr(method="spearman").iloc[0, 1]
    ic_era = mon.loc[ERA:].corr(method="spearman").iloc[0, 1]

    # raw basket stats
    raw = {}
    for leg in ["FL", "KN", "SPY"]:
        raw[leg] = {"pre": perf(b[leg].loc[:ERA - pd.Timedelta(days=1)], leg),
                    "era": perf(b[leg].loc[ERA:], leg)}
    single_era = ((1 + single.loc[ERA:]).prod() - 1).mul(100).round(1).sort_values()

    plac = placebo(spread, len(ev_tab), (1, 10))
    ev_mean = ev_tab.car_p1_p10_pct.mean() / 100
    stats = {
        "spread_cum_pre_pct": round(100 * ((1 + spread.loc[:ERA - pd.Timedelta(days=1)]).prod() - 1), 1),
        "spread_cum_era_pct": round(100 * ((1 + spread.loc[ERA:]).prod() - 1), 1),
        "beta_fl_last": round(bf.dropna().iloc[-1], 2), "beta_kn_last": round(bk.dropna().iloc[-1], 2),
        "fsi_last": round(float(fsi.dropna().iloc[-1]), 2), "fsi_last_date": str(fsi.dropna().index[-1].date()),
        "fsi_components_last": {k: round(float(v), 2) for k, v in comps.ffill().iloc[-1].items()},
        "fsi_ic_full": round(ic_full, 3), "fsi_ic_era": round(ic_era, 3), "fsi_ic_n_months": int(len(mon)),
        "kill_switch_days_active_pct": round(100 * float((kill_switch(fsi) < 1).mean()), 1),
        "event_n": int(len(ev_tab)),
        "event_mean_m1p1_pct": round(ev_tab.car_m1_p1_pct.mean(), 2),
        "event_mean_p1p10_pct": round(ev_tab.car_p1_p10_pct.mean(), 2),
        "event_mean_p1p20_pct": round(ev_tab.car_p1_p20_pct.mean(), 2),
        "event_pos_p1p10": int((ev_tab.car_p1_p10_pct > 0).sum()),
        "event_kn_mean_m1p1_pct": round(ev_tab.knowledge_minus_spy_m1_p1_pct.mean(), 2),
        "event_placebo_pvalue_p1p10": round(float((plac >= ev_mean).mean()), 3),
        "raw_legs": raw,
        "best_single": single_era.tail(5).to_dict(), "worst_single": single_era.head(5).to_dict(),
    }

    # ---------------- figures
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for leg, col, lab in [("FL", FL_COLOR, "Frontline enablers (long)"), ("KN", KN_COLOR, "Knowledge-labour sellers (short)"),
                          ("SPY", INK, "S&P 500")]:
        ax.plot((1 + b[leg].loc[ERA:]).cumprod() * 100, color=col, lw=2, label=lab)
    ax.set_title("Equal-weight baskets since ChatGPT launch (rebased to 100)", loc="left", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(C.FIG / "07_baskets_ai_era.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for col, c in zip(books.columns, [FL_COLOR, "#3a6ea5", "#9b59b6", "#d4a017", INK]):
        ax.plot((1 + books[col]).cumprod(), lw=1.8, color=c, label=col)
    ax.axvline(ERA, color=GREY, ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_title(f"Strategy books, net of {TC_BPS}bp costs, {int(VOL_TARGET*100)}% vol target (log scale)",
                 loc="left", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIG / "08_strategy_equity.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for c in ev_paths:
        ax.plot(ev_paths.index, 100 * ev_paths[c], color=GREY, alpha=0.35, lw=0.8)
    ax.plot(ev_paths.index, 100 * ev_paths.mean(axis=1), color=FL_COLOR, lw=2.5, label="average")
    ax.axvline(0, color=INK, lw=0.8); ax.axhline(0, color=INK, lw=0.6)
    ax.set_title("Hedged frontline-minus-knowledge spread around frontier-AI releases (cum. %, day 0 = release)",
                 loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("trading days"); ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(C.FIG / "09_ai_event_study.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(fsi.loc["2020":], color="#3a6ea5", lw=1.5)
    ax.axhline(0, color=INK, lw=0.6); ax.axvline(ERA, color=GREY, ls="--", lw=1)
    ax.set_title("Frontline Shift Index (z-score; >0 = labour data tilting toward frontline)", loc="left",
                 fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(C.FIG / "10_fsi.png", dpi=150); plt.close(fig)

    perf_tab.to_csv(C.OUT / "strategy_performance.csv", index=False)
    ev_tab.to_csv(C.OUT / "ai_event_study.csv", index=False)
    books.to_csv(C.OUT / "strategy_daily_returns.csv")
    pd.DataFrame({"FSI": fsi}).join(comps).to_csv(C.OUT / "fsi.csv")
    (C.OUT / "strategy_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    return stats, perf_tab, ev_tab


if __name__ == "__main__":
    s, p, e = run()
    pd.set_option("display.width", 200)
    print(json.dumps(s, indent=2, default=str))
    print(p.to_string())
    print(e.to_string())
