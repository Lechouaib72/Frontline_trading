"""Part 1b — What are people *saying* about which jobs will shape the future?

Pipeline on AI-related Reddit / Hacker News comments (see social_collect.py):
  1. Split into sentences; detect occupation mentions (frontline vs knowledge lexicon).
  2. FUTURE-OUTLOOK stance per mention: cues like "AI-proof", "in demand", "shortage"
     (+) vs "replaced", "obsolete", "cooked", "no jobs" (−), with negation handling
     ("won't be replaced" → +).  This is the quantity we care about: not whether the
     writer is happy, but whether they think the job has a future.
  3. General sentiment with a transformer (cardiffnlp/twitter-roberta-base-sentiment-latest)
     on the same sentences, as a cross-check.
  4. CAREER-PIVOT intent: "learn a trade / go to trade school / become an electrician"
     vs "learn to code / bootcamp / get into tech" — revealed *supply-side* leaning.
  5. Monthly indices -> Social Frontline Sentiment Index (SFSI); tests vs official
     labour data (does social lead?) and vs the trading spread (does it predict?),
     plus an event study of outlook around frontier-AI launches.
"""
import html
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .social_collect import load

FL_COLOR, KN_COLOR, GREY, INK = "#1f7a4d", "#b3412c", "#777777", "#222222"
ERA = pd.Timestamp(C.AI_ERA_START)

# ----------------------------------------------------------------------------- lexicons
FRONTLINE_OCC = {
    "Electrician": r"electricians?|lineman|linemen|lineworkers?",
    "Plumber": r"plumb(?:er|ers|ing)",
    "HVAC tech": r"hvac|refrigeration techs?",
    "Welder": r"weld(?:er|ers|ing)",
    "Skilled trades (general)": r"(?:the |skilled |a )trades?|tradesm[ae]n|tradespeople|blue[- ]collar|trade school|apprenticeships?",
    "Carpenter / construction": r"carpent(?:er|ers|ry)|construction workers?|roofers?|masons?|framers?",
    "Mechanic": r"mechanics?|auto techs?|diesel techs?",
    "Nurse": r"nurs(?:e|es|ing)|rns?\b|lpns?",
    "Healthcare (hands-on)": r"paramedics?|emts?|physical therapists?|dental hygienists?|surgeons?|caregivers?|home health aides?|cnas?",
    "Driver / logistics": r"truck ?drivers?|truckers?|delivery drivers?|warehouse workers?",
    "Hospitality / service": r"chefs?|line cooks?|baristas?|waiters?|waitress(?:es)?|bartenders?|hairdressers?|barbers?|fast food workers?",
    "Police / fire": r"firefighters?|police officers?|cops?",
}
KNOWLEDGE_OCC = {
    "Software engineer": r"software (?:engineers?|developers?|engineering|devs?)|programmers?|coders?|web developers?|swes?|developers",
    "Junior dev / entry-level tech": r"junior (?:devs?|developers?|engineers?|roles?)|entry[- ]level (?:tech|dev|swe|software)",
    "Data analyst / scientist": r"data (?:analysts?|scientists?|entry)|business analysts?",
    "Accountant": r"accountants?|accounting|bookkeep(?:er|ers|ing)|cpas?",
    "Lawyer / paralegal": r"lawyers?|paralegals?|attorneys?",
    "Writer / copywriter": r"copywrit(?:er|ers|ing)|content writers?|journalists?|translators?",
    "Designer / artist": r"graphic designers?|illustrators?|artists?|ux designers?",
    "Marketing": r"market(?:er|ers|ing)",
    "Customer service / call centre": r"customer (?:service|support)|call cent(?:er|re)s?|support agents?",
    "Finance / banking": r"financial analysts?|bankers?|investment banking|traders",
    "HR / recruiting": r"\bhr\b|recruit(?:er|ers|ing)|human resources",
    "Admin / office": r"office jobs?|admin(?:istrative)? assistants?|secretar(?:y|ies)|desk jobs?|white[- ]collar",
}

POS = [r"immune", r"never (?:be )?(?:replaced|automated)", r"(?:ai|automation|robot|recession)[- ]?proof", r"future[- ]?proof", r"in (?:high |huge |great )?demand",
       r"job security", r"shortages?", r"always (?:be )?(?:needed|need|in demand)", r"will always",
       r"safe(?:st|r)? (?:from|bet|career|choice)", r"\bsafe\b", r"good money", r"six figures?", r"boom(?:ing)?",
       r"thriv(?:e|ing)", r"secure", r"lucrative", r"hiring like crazy", r"can'?t find (?:enough|workers|people)",
       r"(?:ai|robots?|chatgpt|llms?) (?:can'?t|cannot|won'?t|will never) (?:do|fix|wire|crawl|replace|climb|touch|repair)",
       r"hard to automate", r"great career", r"growing"]
NEG = [r"replac(?:e|ed|es|ing)\b", r"replacement for", r"obsolete", r"automat(?:e|ed|ing)", r"\bdead\b", r"dying", r"doomed",
       r"(?:over)?saturated", r"laid off", r"layoffs?", r"no jobs?", r"can'?t find (?:a |any )?(?:job|work)",
       r"wiped out", r"eliminat(?:e|ed|ing)", r"redundant", r"outsourc(?:e|ed|ing)", r"cooked", r"\btoast\b",
       r"\bfucked\b", r"screwed", r"disappear(?:ing)?", r"gone in", r"race to the bottom", r"worthless", r"bleak",
       r"won'?t exist", r"taking (?:our|their|the) jobs", r"take (?:our|their|the|your) jobs?"]
NEGATORS = (r"(?:not|n't|never|no|cannot|can'?t|won'?t|hard to|unlikely to|impossible to|nowhere near|"
            r"far from|rather than|instead of|less likely|least likely|safe from|immune to|resistant to|"
            r"(?:hard|difficult)(?: for (?:me|us))? to (?:see|imagine)|doubt|don'?t think|do not think|don'?t believe|skeptical|last to be)")

PIVOT_FRONTLINE = (r"(?:learn(?:ing)?|go(?:ing)? into|get(?:ting)? into|switch(?:ing|ed)? to|pivot(?:ing|ed)? to|"
                   r"start(?:ing|ed)?|join(?:ing|ed)?|enroll(?:ing|ed)? in|went to|go(?:ing)? to|consider(?:ing)?|"
                   r"become|becoming|apply(?:ing)? (?:for|to)|look(?:ing)? into)\s+(?:a |an |the )?"
                   r"(?:trades?\b|trade school|apprenticeship|electrician|plumb|hvac|welding|lineman|nursing school|nursing|"
                   r"electrical apprentice|union apprentice)")
PIVOT_KNOWLEDGE = (r"(?:learn(?:ing)? (?:to code|programming|python|javascript|web dev)|coding bootcamp|bootcamp|"
                   r"(?:get(?:ting)?|break(?:ing)?|go(?:ing)?|switch(?:ing)?|pivot(?:ing)?) (?:in)?to (?:tech|software|"
                   r"data science|programming|cs)\b|cs degree|computer science degree|major(?:ing)? in (?:cs|computer science))")

_FL = {k: re.compile(rf"\b(?:{v})\b", re.I) for k, v in FRONTLINE_OCC.items()}
_KN = {k: re.compile(rf"\b(?:{v})\b", re.I) for k, v in KNOWLEDGE_OCC.items()}
_POS = re.compile(rf"(?:{'|'.join(POS)})", re.I)
_NEG = re.compile(rf"(?:{'|'.join(NEG)})", re.I)
_NEGATED_NEG = re.compile(rf"{NEGATORS}\W+(?:[\w']+\W+){{0,6}}?(?:{'|'.join(NEG)})", re.I)
_NEGATED_POS = re.compile(rf"{NEGATORS}\W+(?:\w+\W+){{0,2}}?(?:{'|'.join(POS[:6])})", re.I)
_PIV_FL = re.compile(PIVOT_FRONTLINE, re.I)
_PIV_KN = re.compile(PIVOT_KNOWLEDGE, re.I)
_AI = re.compile(r"\b(?:ai|a\.i\.|chatgpt|gpt-?\d|llms?|claude|gemini|copilot|automation|robots?|agents?)\b", re.I)
_CTX = re.compile(r"\b(?:jobs?|careers?|future|demand|hir(?:e|ed|ing)|employ\w*|work(?:ers)?|"
                  r"unemploy\w*|layoffs?|laid off|salar\w+|wages?|pay|field|industry|profession)\b", re.I)
_JOBAD = re.compile(r"\|\s*(?:full[- ]?time|part[- ]?time|remote|onsite|on-site|hybrid|contract)\b|who'?s hiring", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_TAG = re.compile(r"<[^>]+>")


def clean(t):
    t = html.unescape(_TAG.sub(" ", t or ""))
    t = re.sub(r"https?://\S+", " ", t)
    return re.sub(r"[ \t]+", " ", t).strip()


def outlook(sent):
    """+1 / -1 / 0 future-outlook stance of a sentence."""
    pos = len(_POS.findall(sent))
    neg = len(_NEG.findall(sent))
    nn = len(_NEGATED_NEG.findall(sent))      # "won't be replaced" -> flips neg to pos
    npos = len(_NEGATED_POS.findall(sent))    # "not in demand" -> flips pos to neg
    pos, neg = pos - npos + nn, neg - nn + npos
    return int(np.sign(pos - neg))


_CLAUSE = re.compile(r"([,;:]|\b(?:but|while|whereas|although|though|however|before|than|unlike|except)\b)", re.I)
_CONTRAST = {"before", "than", "unlike", "except"}


def occ_outlook(sent, rx):
    """Stance toward the occupation matched by rx.

    Its own clause first; if that clause has no cue, inherit the sentence stance —
    inverted when the occupation sits after a contrast word ("replace architects
    before plumbers", "more likely than nurses")."""
    parts = _CLAUSE.split(sent)
    prev_sep = ""
    for i, c in enumerate(parts):
        if i % 2 == 1:
            prev_sep = c.strip().lower()
            continue
        if rx.search(c):
            o = outlook(c)
            if o != 0:
                return o
            o = outlook(sent)
            return -o if prev_sep in _CONTRAST else o
    return outlook(sent)


def mentions(df):
    """One row per (comment, sentence, occupation) mention in an AI-context comment."""
    rows = []
    for r in df.itertuples():
        text = clean(r.text)
        if not _AI.search(text) or _JOBAD.search(text):
            continue
        for s in _SENT.split(text):
            if len(s) < 15 or len(s) > 600 or not (_AI.search(s) or _CTX.search(s)):
                continue
            for grp, lex in (("frontline", _FL), ("knowledge", _KN)):
                for occ, rx in lex.items():
                    if rx.search(s):
                        rows.append((r.src, r.id, r.date, r.community, grp, occ, s, occ_outlook(s, rx)))
    return pd.DataFrame(rows, columns=["src", "id", "date", "community", "group", "occupation", "sentence", "outlook"])


def pivots(df):
    t = df.text.map(clean)
    out = df[["src", "id", "date", "community"]].copy()
    out["pivot_fl"] = t.str.contains(_PIV_FL)
    out["pivot_kn"] = t.str.contains(_PIV_KN)
    out["ai_ctx"] = t.str.contains(_AI)
    return out


def transformer_sentiment(sentences, batch=64):
    """Positive minus negative probability, cardiffnlp twitter-roberta (cached by sentence hash)."""
    cache_p = C.RAW / "social" / "roberta_cache.parquet"
    cache = pd.read_parquet(cache_p) if cache_p.exists() else pd.DataFrame(columns=["h", "s"])
    cmap = dict(zip(cache.h, cache.s))
    hs = sentences.map(lambda x: str(hash(x) & 0xFFFFFFFFFFFF))
    todo = sorted(set(zip(hs, sentences)) - {(h, None) for h in []})
    todo = [(h, s) for h, s in todo if h not in cmap]
    if todo:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        tok = AutoTokenizer.from_pretrained(name)
        mod = AutoModelForSequenceClassification.from_pretrained(name).eval()
        torch.set_num_threads(max(1, torch.get_num_threads()))
        new = []
        with torch.no_grad():
            for i in range(0, len(todo), batch):
                chunk = todo[i:i + batch]
                enc = tok([s for _, s in chunk], truncation=True, max_length=128, padding=True, return_tensors="pt")
                p = mod(**enc).logits.softmax(-1).numpy()      # [neg, neu, pos]
                new += [(h, float(pp[2] - pp[0])) for (h, _), pp in zip(chunk, p)]
                if (i // batch) % 50 == 0:
                    print(f"  roberta {i}/{len(todo)}", flush=True)
        cache = pd.concat([cache, pd.DataFrame(new, columns=["h", "s"])])
        cache.to_parquet(cache_p)
        cmap = dict(zip(cache.h, cache.s))
    return hs.map(cmap)


# ----------------------------------------------------------------------------- indices
NEUTRAL_HN = ["hn:AI jobs", "hn:AI replace"]


def monthly_indices(m, pv):
    m = m.copy()
    m["month"] = m.date.dt.to_period("M").dt.to_timestamp()
    g = m.groupby(["month", "group"])
    # share of voice only from occupation-neutral sources (HN queries that name an
    # occupation would mechanically inflate that occupation's share)
    neutral = m[~m.community.str.startswith("hn:") | m.community.isin(NEUTRAL_HN)]
    cnt = neutral.groupby(["month", "group"]).size().unstack().fillna(0)
    net = g.outlook.mean().unstack()               # mean stance in [-1, 1]
    sent = g.roberta.mean().unstack()
    idx = pd.DataFrame({
        "share_of_voice_fl": cnt.frontline / cnt.sum(axis=1),
        "outlook_fl": net.frontline, "outlook_kn": net.knowledge,
        "outlook_gap": net.frontline - net.knowledge,
        "roberta_fl": sent.frontline, "roberta_kn": sent.knowledge,
        "roberta_gap": sent.frontline - sent.knowledge,
        "n_mentions": g.size().unstack().fillna(0).sum(axis=1),
    })
    pv = pv.copy()
    pv["month"] = pv.date.dt.to_period("M").dt.to_timestamp()
    career = pv[pv.community.isin(["careerguidance", "findapath", "jobs", "cscareerquestions"])]
    pg = career.groupby("month")
    idx["pivot_fl_per_1k"] = 1000 * pg.pivot_fl.mean()
    idx["pivot_kn_per_1k"] = 1000 * pg.pivot_kn.mean()
    s = pg.pivot_fl.sum() + pg.pivot_kn.sum()
    idx["pivot_ratio_fl"] = pg.pivot_fl.sum() / s.replace(0, np.nan)
    return idx.loc[: pd.Timestamp("2026-11-01")]


def sfsi(idx):
    """Social Frontline Sentiment Index: mean of expanding-z of three monthly gauges, 3m smoothed."""
    cols = ["share_of_voice_fl", "outlook_gap", "pivot_ratio_fl"]
    sm = idx[cols].rolling(3, min_periods=2).mean()
    z = (sm - sm.expanding(6).mean()) / sm.expanding(6).std()
    return z.mean(axis=1).rename("SFSI"), z


# ----------------------------------------------------------------------------- tests
def lead_lag_vs_official(s):
    """Does the social index lead the official labour data?  corr(ΔSFSI_t, Δofficial_{t+k})."""
    fsi = pd.read_csv(C.OUT / "fsi.csv", index_col=0, parse_dates=True)
    off = fsi[["indeed", "ces"]].resample("MS").last()
    ds = s.diff()
    out = {}
    for col in off:
        d = off[col].diff()
        out[col] = {k: round(ds.corr(d.shift(-k)), 3) for k in [-3, -1, 0, 1, 3, 6]}
    return out


def predictive_test(s):
    """Rank IC of SFSI (known at month end) vs next-month and next-3-month hedged spread."""
    from . import data as D
    from .strategy import basket_returns, hedged_spread
    b, _ = basket_returns(D.prices())
    sp, _, _ = hedged_spread(b)
    m1 = sp.resample("MS").sum()
    f1 = m1.shift(-1)
    f3 = m1[::-1].rolling(3).sum()[::-1].shift(-1)
    df = pd.concat([s, f1.rename("f1"), f3.rename("f3"), m1.rename("cur")], axis=1).dropna()
    era = df.loc[ERA:]
    return {"n_months": int(len(df)),
            "ic_1m_full": round(df.corr("spearman").loc["SFSI", "f1"], 3),
            "ic_3m_full": round(df.corr("spearman").loc["SFSI", "f3"], 3),
            "ic_1m_era": round(era.corr("spearman").loc["SFSI", "f1"], 3),
            "ic_3m_era": round(era.corr("spearman").loc["SFSI", "f3"], 3),
            "corr_contemporaneous_era": round(era.corr("spearman").loc["SFSI", "cur"], 3)}


def ai_event_reaction(m, days=14):
    """Outlook of knowledge vs frontline mentions in the 14 days after vs before each AI launch."""
    rows = []
    for d, name in C.AI_EVENTS.items():
        d = pd.Timestamp(d)
        pre = m[(m.date >= d - pd.Timedelta(days=days)) & (m.date < d)]
        post = m[(m.date >= d) & (m.date < d + pd.Timedelta(days=days))]
        if len(pre) < 30 or len(post) < 30:
            continue
        r = {"date": str(d.date()), "event": name}
        for g in ["frontline", "knowledge"]:
            r[f"{g}_outlook_chg"] = round(post[post.group == g].outlook.mean() - pre[pre.group == g].outlook.mean(), 3)
        r["sov_fl_chg"] = round((post.group == "frontline").mean() - (pre.group == "frontline").mean(), 3)
        r["n_pre"], r["n_post"] = len(pre), len(post)
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- run
def run(use_transformer=True):
    C.FIG.mkdir(parents=True, exist_ok=True)
    df = load()
    df = df[df.date >= "2021-01-01"]
    m = mentions(df)
    m["roberta"] = transformer_sentiment(m.sentence) if use_transformer else np.nan
    pv = pivots(df)
    idx = monthly_indices(m, pv)
    s, z = sfsi(idx)

    era = m[m.date >= ERA]
    pre = m[m.date < ERA]
    neu = lambda x: x[~x.community.str.startswith("hn:") | x.community.isin(NEUTRAL_HN)]
    occ = (era.groupby(["group", "occupation"])
              .agg(mentions=("outlook", "size"), net_outlook=("outlook", "mean"),
                   pct_positive=("outlook", lambda x: (x > 0).mean()), pct_negative=("outlook", lambda x: (x < 0).mean()),
                   roberta=("roberta", "mean"))
              .reset_index())
    occ_pre = pre.groupby("occupation").outlook.mean().rename("net_outlook_pre_chatgpt")
    occ = occ.merge(occ_pre, on="occupation", how="left").sort_values("net_outlook", ascending=False)
    occ = occ[occ.mentions >= 50]

    ev = ai_event_reaction(m)
    # validation sample for audit (stratified by stance)
    nz = m[m.outlook != 0]
    sample = pd.concat([nz[nz.outlook == v].sample(min(40, (nz.outlook == v).sum()), random_state=1) for v in (-1, 1)]
                       + [m[m.outlook == 0].sample(min(20, (m.outlook == 0).sum()), random_state=1)]
                       )[["date", "group", "occupation", "outlook", "sentence"]]

    pv_era, pv_pre = pv[pv.date >= ERA], pv[pv.date < ERA]
    career = ["careerguidance", "findapath", "jobs", "cscareerquestions"]
    def piv(x):
        x = x[x.community.isin(career)]
        return {"comments": int(len(x)), "trade_pivot_per_1k": round(1000 * x.pivot_fl.mean(), 2),
                "code_pivot_per_1k": round(1000 * x.pivot_kn.mean(), 2),
                "trade_share_of_pivots": round(x.pivot_fl.sum() / max(1, x.pivot_fl.sum() + x.pivot_kn.sum()), 3)}

    stats = {
        "n_comments": int(len(df)), "n_reddit": int((df.src == "reddit").sum()), "n_hn": int((df.src == "hn").sum()),
        "n_mentions": int(len(m)), "date_min": str(df.date.min().date()), "date_max": str(df.date.max().date()),
        "share_of_voice_fl_pre": round((neu(pre).group == "frontline").mean(), 3),
        "share_of_voice_fl_era": round((neu(era).group == "frontline").mean(), 3),
        "outlook_fl_pre": round(pre[pre.group == "frontline"].outlook.mean(), 3),
        "outlook_fl_era": round(era[era.group == "frontline"].outlook.mean(), 3),
        "outlook_kn_pre": round(pre[pre.group == "knowledge"].outlook.mean(), 3),
        "outlook_kn_era": round(era[era.group == "knowledge"].outlook.mean(), 3),
        "roberta_fl_era": round(era[era.group == "frontline"].roberta.mean(), 3),
        "roberta_kn_era": round(era[era.group == "knowledge"].roberta.mean(), 3),
        "pivots_pre": piv(pv_pre), "pivots_era": piv(pv_era),
        "pivots_last12m": piv(pv[pv.date >= pv.date.max() - pd.Timedelta(days=365)]),
        "sfsi_last": round(float(s.dropna().iloc[-1]), 2), "sfsi_last_month": str(s.dropna().index[-1].date()),
        "lead_lag_vs_official": lead_lag_vs_official(s),
        "predictive": predictive_test(s),
        "ai_event_kn_outlook_chg_mean": round(ev.knowledge_outlook_chg.mean(), 3) if len(ev) else None,
        "ai_event_fl_outlook_chg_mean": round(ev.frontline_outlook_chg.mean(), 3) if len(ev) else None,
        "ai_event_kn_outlook_worse_count": int((ev.knowledge_outlook_chg < 0).sum()) if len(ev) else None,
        "ai_event_n": int(len(ev)),
    }

    # ---------------- figures
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax = axes[0]
    ax.plot(idx.outlook_fl.rolling(3).mean(), color=FL_COLOR, lw=2, label="Frontline jobs")
    ax.plot(idx.outlook_kn.rolling(3).mean(), color=KN_COLOR, lw=2, label="Knowledge jobs")
    ax.axhline(0, color=INK, lw=0.6); ax.axvline(ERA, color=GREY, ls="--", lw=1)
    ax.set_title("What people say about a job's future in AI discussions\n(net outlook: share 'safe / in demand' minus "
                 "share 'replaced / cooked', 3m avg)", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
    ax = axes[1]
    ax.plot(100 * idx.share_of_voice_fl.rolling(3).mean(), color=FL_COLOR, lw=2)
    ax.axvline(ERA, color=GREY, ls="--", lw=1)
    ax.set_title("Frontline share of all occupation mentions in AI discussions, % (3m avg)", loc="left",
                 fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(C.FIG / "11_social_outlook.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(idx.pivot_fl_per_1k.rolling(3).mean(), color=FL_COLOR, lw=2, label="'learn a trade / trade school / become an electrician…'")
    ax.plot(idx.pivot_kn_per_1k.rolling(3).mean(), color=KN_COLOR, lw=2, label="'learn to code / bootcamp / get into tech…'")
    ax.axvline(ERA, color=GREY, ls="--", lw=1)
    ax.set_title("Career-pivot talk in r/careerguidance, r/findapath, r/jobs, r/cscareerquestions\n(per 1,000 AI-related comments, 3m avg)",
                 loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(C.FIG / "12_social_pivots.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    o = occ.sort_values("net_outlook")
    ax.barh(o.occupation, o.net_outlook, color=[FL_COLOR if g == "frontline" else KN_COLOR for g in o.group])
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_title("Which jobs do people think have a future? Net outlook since ChatGPT\n(green = frontline, red = knowledge; ≥50 mentions)",
                 loc="left", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(C.FIG / "13_social_occupation_ranking.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(s, color="#3a6ea5", lw=1.8)
    ax.axhline(0, color=INK, lw=0.6); ax.axvline(ERA, color=GREY, ls="--", lw=1)
    ax.set_title("Social Frontline Sentiment Index (SFSI, z-score)", loc="left", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(C.FIG / "14_sfsi.png", dpi=150); plt.close(fig)

    idx.join(s).join(z.add_prefix("z_")).to_csv(C.OUT / "social_monthly.csv")
    occ.to_csv(C.OUT / "social_occupation_ranking.csv", index=False)
    ev.to_csv(C.OUT / "social_ai_event_reaction.csv", index=False)
    sample.to_csv(C.OUT / "social_validation_sample.csv", index=False)
    (C.OUT / "social_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    return stats, occ, ev


if __name__ == "__main__":
    import sys
    st, occ, ev = run(use_transformer="--no-transformer" not in sys.argv)
    pd.set_option("display.width", 200)
    print(json.dumps(st, indent=2, default=str))
    print(occ.to_string())
    print(ev.to_string())
