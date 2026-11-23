"""Part 1 — Is the labour market really leaning into frontline work?

Five independent lenses, each comparing frontline vs knowledge work:
  1. Indeed postings (demand, daily)            -> composite + skill-matched pairs
  2. Difference-in-differences on postings      -> is the gap an *AI-era* break?
  3. BLS payrolls (realised employment)         -> frontline share of jobs
  4. BLS wages                                   -> relative price of frontline labour
  5. Young graduate unemployment                 -> the entry-level white-collar squeeze
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from . import config as C
from . import data as D

FL_COLOR, KN_COLOR, GREY = "#1f7a4d", "#b3412c", "#777777"
ERA = pd.Timestamp(C.AI_ERA_START)


def _style(ax, title, ylabel=""):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    ax.axvline(ERA, color=GREY, ls="--", lw=1)
    ax.annotate("ChatGPT", (ERA, ax.get_ylim()[1]), fontsize=8, color=GREY,
                ha="left", va="top", xytext=(3, -3), textcoords="offset points")


def composite(df, cols):
    """Equal-weight geometric mean of indices (so no single category dominates)."""
    return np.exp(np.log(df[cols]).mean(axis=1))


def indeed_lens(ind):
    fl = composite(ind, list(C.INDEED_FRONTLINE))
    kn = composite(ind, list(C.INDEED_KNOWLEDGE))
    out = pd.DataFrame({"frontline": fl, "knowledge": kn})
    out["spread"] = np.log(fl / kn)
    m = out.resample("ME").mean()
    at_era = m.loc[:ERA].iloc[-1]
    last = m.iloc[-1]
    stats = {
        "indeed_last_date": str(ind.index.max().date()),
        "indeed_frontline_vs_feb2020": round(last.frontline - 100, 1),
        "indeed_knowledge_vs_feb2020": round(last.knowledge - 100, 1),
        "indeed_frontline_since_chatgpt_pct": round(100 * (last.frontline / at_era.frontline - 1), 1),
        "indeed_knowledge_since_chatgpt_pct": round(100 * (last.knowledge / at_era.knowledge - 1), 1),
        "indeed_ratio_now": round(last.frontline / last.knowledge, 2),
    }
    cat = ind.resample("ME").mean()
    chg = (100 * (cat.iloc[-1] / cat.loc[:ERA].iloc[-1] - 1)).sort_values()
    pairs = []
    for f, k, label in C.MATCHED_PAIRS:
        r = cat[f] / cat[k]
        pairs.append({"label": label, "frontline": C.INDEED_FRONTLINE[f], "knowledge": C.INDEED_KNOWLEDGE[k],
                      "ratio_at_chatgpt": round(r.loc[:ERA].iloc[-1], 2), "ratio_now": round(r.iloc[-1], 2),
                      "rel_change_pct": round(100 * (r.iloc[-1] / r.loc[:ERA].iloc[-1] - 1), 1)})
    stats["matched_pairs"] = pairs
    stats["pairs_all_positive"] = all(p["rel_change_pct"] > 0 for p in pairs)

    # figure: composites
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(out.frontline.rolling(7).mean(), color=FL_COLOR, lw=2, label="Frontline occupations (15)")
    ax.plot(out.knowledge.rolling(7).mean(), color=KN_COLOR, lw=2, label="Knowledge occupations (17)")
    ax.axhline(100, color="black", lw=0.6)
    _style(ax, "Indeed job postings, index Feb-2020 = 100", "index")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(C.FIG / "01_indeed_composites.png", dpi=150); plt.close(fig)

    # figure: category bar chart
    fig, ax = plt.subplots(figsize=(9, 7))
    names = {**C.INDEED_FRONTLINE, **C.INDEED_KNOWLEDGE}
    colors = [FL_COLOR if c in C.INDEED_FRONTLINE else KN_COLOR for c in chg.index]
    ax.barh([names[c] for c in chg.index], chg.values, color=colors)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_title("Change in Indeed postings since ChatGPT launch (Nov-2022 → latest), %",
                 loc="left", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(C.FIG / "02_indeed_categories.png", dpi=150); plt.close(fig)
    return out, chg, stats


def patsy_env():
    """Formula namespace without our `C` (config) shadowing patsy's C()."""
    import patsy
    return patsy.EvalEnvironment([{"Q": patsy.builtins.Q, "C": patsy.builtins.C, "np": np}])


def did_lens(ind):
    """Panel DiD on monthly log postings, category & month fixed effects.

    log(P_ct) = a_c + g_t + sum_q b_q * Knowledge_c * 1[t in q] + e
    b_q traced by half-year relative to 2022H2 (omitted).  If the knowledge
    discount were just the 2022 rate shock we'd expect it to open in 2022H1;
    a break at/after 2022H2 that keeps widening points to the AI era.
    """
    m = np.log(ind.resample("ME").mean()).stack().rename("lp").reset_index()
    m.columns = ["date", "cat", "lp"]
    m = m.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    m["kn"] = m["cat"].isin(C.INDEED_KNOWLEDGE).astype(int)
    m["post"] = (m["date"] >= ERA).astype(int)
    m["hy"] = m["date"].dt.year.astype(str) + "H" + np.where(m["date"].dt.month <= 6, "1", "2")
    simple = smf.ols("lp ~ C(cat) + C(date) + kn:post", data=m, eval_env=patsy_env()).fit(
        cov_type="cluster", cov_kwds={"groups": m["cat"]})
    beta = simple.params["kn:post"]
    levels = sorted(m["hy"].unique())
    base = "2022H2"
    for h in levels:
        if h != base:
            m[f"d_{h}"] = ((m["hy"] == h) & (m["kn"] == 1)).astype(int)
    terms = " + ".join(f"Q('d_{h}')" for h in levels if h != base)
    es = smf.ols(f"lp ~ C(cat) + C(date) + {terms}", data=m, eval_env=patsy_env()).fit(
        cov_type="cluster", cov_kwds={"groups": m["cat"]})
    rows = []
    for h in levels:
        if h == base:
            rows.append((h, 0.0, 0.0, 0.0))
        else:
            k = f"Q('d_{h}')"
            lo, hi = es.conf_int().loc[k]
            rows.append((h, es.params[k], lo, hi))
    ev = pd.DataFrame(rows, columns=["half", "b", "lo", "hi"])
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(ev))
    ax.errorbar(x, 100 * ev.b, yerr=[100 * (ev.b - ev.lo), 100 * (ev.hi - ev.b)],
                fmt="o", color=KN_COLOR, capsize=3)
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(levels.index(base), color=GREY, ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(ev.half, rotation=45, fontsize=8)
    ax.set_title("Event-study DiD: knowledge-occupation postings vs frontline (log pts ×100, 95% CI)\n"
                 "category & month fixed effects, clustered by category; base = 2022H2",
                 loc="left", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(C.FIG / "03_did_event_study.png", dpi=150); plt.close(fig)
    return ev, {"did_beta_logpts": round(beta, 3), "did_beta_pct": round(100 * (np.exp(beta) - 1), 1),
                "did_tstat": round(simple.tvalues["kn:post"], 2),
                "did_pre_2022H1": round(float(ev.set_index("half").loc["2022H1", "b"]), 3),
                "did_latest": round(float(ev.b.iloc[-1]), 3)}


def ces_lens(f):
    fl = f[list(C.CES_FRONTLINE)].dropna()
    kn = f[list(C.CES_KNOWLEDGE)].dropna()
    idx = fl.index.intersection(kn.index)
    # avoid double counting nested series: top-level only for the share
    fl_top = f[["USCONS", "CES6562000001", "CES4300000001", "USLAH", "MANEMP"]].loc[idx].sum(axis=1)
    kn_top = f[["USINFO", "CES6054000001", "CES5552000001", "CES6055000001"]].loc[idx].sum(axis=1)
    share = (100 * fl_top / (fl_top + kn_top)).loc["2010":]
    def ann(s, a, b):
        s = s.dropna(); a, b = s.loc[:a].iloc[-1], s.loc[:b].iloc[-1]
        return a, b
    rows = []
    last = idx.max()
    yrs_post = (last - ERA).days / 365.25
    for code, name in {**C.CES_FRONTLINE, **C.CES_KNOWLEDGE}.items():
        s = f[code].dropna()
        pre = 100 * ((s.loc[:"2019-12-31"].iloc[-1] / s.loc[:"2015-12-31"].iloc[-1]) ** (1 / 4) - 1)
        post = 100 * ((s.iloc[-1] / s.loc[:ERA].iloc[-1]) ** (1 / yrs_post) - 1)
        rows.append({"series": code, "industry": name,
                     "group": "frontline" if code in C.CES_FRONTLINE else "knowledge",
                     "cagr_2015_19": round(pre, 2), "cagr_since_chatgpt": round(post, 2),
                     "change_k": round(s.iloc[-1] - s.loc[:ERA].iloc[-1], 0)})
    tab = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(share, color=FL_COLOR, lw=2)
    _style(ax, "Frontline share of jobs: construction + health care + transport/warehousing +\n"
               "leisure/hospitality + manufacturing, as % of those + info/prof-tech/finance/mgmt", "%")
    fig.tight_layout(); fig.savefig(C.FIG / "04_ces_frontline_share.png", dpi=150); plt.close(fig)
    fl_add = tab.query("group=='frontline' and series in ['USCONS','CES6562000001','CES4300000001','USLAH','MANEMP']").change_k.sum()
    kn_add = tab.query("group=='knowledge' and series in ['USINFO','CES6054000001','CES5552000001','CES6055000001']").change_k.sum()
    return tab, {"ces_last": str(last.date()), "ces_share_at_chatgpt": round(share.loc[:ERA].iloc[-1], 2),
                 "ces_share_now": round(share.iloc[-1], 2),
                 "ces_frontline_jobs_added_k": int(fl_add), "ces_knowledge_jobs_added_k": int(kn_add),
                 "computer_systems_design_change_k": int(tab.set_index("series").loc["CES6054150001", "change_k"])}


def wage_lens(f):
    fl = f[list(C.AHE_FRONTLINE)].dropna()
    kn = f[list(C.AHE_KNOWLEDGE)].dropna()
    g_fl = (np.log(fl).diff(12).mean(axis=1) * 100)
    g_kn = (np.log(kn).diff(12).mean(axis=1) * 100)
    rel = (np.log(fl).mean(axis=1) - np.log(kn).mean(axis=1))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(g_fl.loc["2015":], color=FL_COLOR, lw=2, label="Frontline industries")
    ax.plot(g_kn.loc["2015":], color=KN_COLOR, lw=2, label="Knowledge industries")
    _style(ax, "Average hourly earnings growth, y/y %", "%")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(C.FIG / "05_wage_growth.png", dpi=150); plt.close(fig)
    post = g_fl.loc[ERA:] - g_kn.loc[ERA:]
    return {"wage_gap_avg_since_chatgpt_pp": round(post.mean(), 2),
            "wage_rel_level_change_pct": round(100 * (np.exp(rel.iloc[-1] - rel.loc[:ERA].iloc[-1]) - 1), 2),
            "wage_months_frontline_faster_pct": round(100 * (post > 0).mean(), 0)}


def grad_lens(f):
    g = f[["CGBD2024", "LNS14000036", "LNS14027662", "LNS14027660", "UNRATE"]].dropna().rolling(12).mean()
    gap = g["CGBD2024"] - g["LNS14000036"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(gap.loc["2005":], color=KN_COLOR, lw=2)
    ax.axhline(0, color="black", lw=0.6)
    _style(ax, "Young-graduate penalty: unemployment of bachelor's 20-24 minus all 20-24\n"
               "(pp, 12m average; above zero = a degree no longer protects new entrants)", "pp")
    fig.tight_layout(); fig.savefig(C.FIG / "06_young_grad_gap.png", dpi=150); plt.close(fig)
    return {"grad_gap_now_pp": round(gap.iloc[-1], 2), "grad_gap_at_chatgpt_pp": round(gap.loc[:ERA].iloc[-1], 2),
            "grad_gap_2010s_avg_pp": round(gap.loc["2012":"2019"].mean(), 2),
            "grad_gap_last": str(gap.index[-1].date())}


def run():
    C.FIG.mkdir(parents=True, exist_ok=True)
    ind = D.indeed()
    f = D.fred([*C.CES_FRONTLINE, *C.CES_KNOWLEDGE, *C.AHE_FRONTLINE, *C.AHE_KNOWLEDGE, *C.OTHER_FRED])
    comp, chg, s1 = indeed_lens(ind)
    ev, s2 = did_lens(ind)
    tab, s3 = ces_lens(f)
    s4 = wage_lens(f)
    s5 = grad_lens(f)
    stats = {**s1, **s2, **s3, **s4, **s5}
    tab.to_csv(C.OUT / "ces_table.csv", index=False)
    chg.rename("pct_change_since_chatgpt").to_csv(C.OUT / "indeed_category_changes.csv")
    ev.to_csv(C.OUT / "did_event_study.csv", index=False)
    (C.OUT / "evidence_stats.json").write_text(json.dumps(stats, indent=2))
    return stats, comp


if __name__ == "__main__":
    s, _ = run()
    print(json.dumps(s, indent=2))
