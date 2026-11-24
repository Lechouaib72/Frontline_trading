# The Frontline Shift — an event-driven AI-labour trade

**View:** frontier AI is substituting for screen-based *knowledge* labour faster than anyone can
automate *frontline* labour (hands-on, in-person: trades, care, logistics, service). Labour demand,
training and spending migrate toward the frontline. Equities that **sell knowledge-labour hours**
are structurally impaired, while those that **train, equip and deploy frontline workers** gain.

Everything here is reproducible: `pip install -r requirements.txt && python run_all.py`
(add `--refresh` to re-download). Data through **2026-11-23** (prices), **2026-11-13** (Indeed),
**Oct-2026** (BLS).

---

## Part 1 — Is the labour market really leaning into frontline work?

Six independent lenses, frontline vs knowledge in each (five from official data, one from social media).

| Lens | Result | Supports thesis? |
|---|---|---|
| **Indeed postings** (32 occupations) | Frontline composite **+13% vs Feb-2020**; knowledge **−11%**. Frontline/knowledge ratio 1.27 | ✅ Strong |
| **Skill-matched pairs** (same wage tier, different AI exposure) | All 5 pairs moved toward frontline since ChatGPT: med-technician vs med-records **+33%**, logistics support vs documentation **+32%**, construction vs architecture **+26%**, nursing vs HR **+7%**, food service vs customer service **+3%** | ✅ Strong — rules out "it's just skill/pay level" |
| **Diff-in-diff on postings** (category + month FE, clustered SEs) | Knowledge postings **−19% relative** post-ChatGPT (t = −3.5). Event study: gap opens 2022H2→2023H1 and *never closes*, even through 2024-25 rate cuts | ✅ Strong, with caveat ↓ |
| **BLS payrolls** (realised jobs, since Nov-2022) | Frontline industries **+4.16M** jobs; knowledge industries **−180k** (Information −381k, computer systems design −129k, employment services −588k). Frontline share 73.7% → 75.0% | ✅ Strong (health care is 3.1M of it) |
| **Young-graduate penalty** (unemp. bachelor's 20-24 minus all 20-24) | 2010s avg **−3.5pp** → **−0.8pp** today: a degree barely protects new entrants anymore | ✅ Supportive |
| **Wages** (avg hourly earnings) | Frontline wage growth has **not** outpaced knowledge (−0.2pp/yr avg since ChatGPT) | ❌ Not yet |

**Honest caveats.**
1. *Rate shock confound:* ~⅓ of the postings gap opened in 2022H2 — the Fed hiking cycle and the
   tech over-hiring hangover, before ChatGPT. What's AI-like is that the gap **persisted and widened
   after rates fell**. See `outputs/figures/03_did_event_study.png`.
2. *Health care = demographics:* ex-health care, frontline still added ~1.1M jobs vs −180k knowledge.
   Transport is flat and manufacturing is shrinking: the shift is **"care + skilled trades"**, not "all blue collar".
3. *Wages:* a quantity shift without a price shift. Partly composition (cutting junior knowledge
   roles mechanically lifts the average knowledge wage), partly frontline supply. **This is the
   un-priced part of the thesis** — if frontline wage growth breaks above knowledge, it's the next leg
   (and it *hurts* employers of frontline labour; see trade 6 below).

### Lens 6 — What people are saying: social-media sentiment (Reddit + Hacker News)

**157,546 public comments** (Jan-2021 → Nov-2026): Reddit (r/careerguidance, r/findapath,
r/cscareerquestions, r/electricians, r/nursing, r/Futurology; comments mentioning "AI", first 7 days
of each month, via the Arctic Shift archive) and Hacker News (Algolia; 8 job/AI queries). No author
handles stored. **23,444 occupation mentions** in 24 occupations, each scored for *future-outlook
stance* ("AI-proof / in demand / shortage" vs "replaced / cooked / laid off"), with negation, clause
and contrast handling ("AI will replace architects *before* plumbers" → plumbers positive).

| Social gauge | Pre-ChatGPT | AI era | Read |
|---|---|---|---|
| Net outlook, **knowledge** jobs | −0.057 | **−0.117** | Talk about knowledge jobs' future roughly doubled in negativity |
| Net outlook, **frontline** jobs | −0.024 | **−0.006** | Flat-to-better |
| Outlook gap (frontline − knowledge) | +0.03 | **+0.11** | Change **+0.09, 95% CI [0.02, 0.14]** (comment-clustered bootstrap); gap ≥ +0.08 every year 2023-26 |
| Generic sentiment (RoBERTa, independent model) | FL −0.23 / KN −0.10 | FL −0.18 / KN −0.21 | Same direction with a different method: the gap flipped sign |
| Career-pivot talk in career subs ("learn a trade" vs "learn to code") | 0 trade vs 75 code per 1k comments | **6.9 trade vs 26 code** | Trades went from 0% to **21% of pivot talk** (23% last 12m); "learn to code" fell by ⅔ |
| Frontline share of occupation mentions | 9.8% | 14.7% | Mostly Reddit (13.5% → 30.6%); HN only 8.2% → 9.2% |

**Which jobs do people think will shape the future?** (net outlook since ChatGPT, ≥50 mentions,
`outputs/figures/13_social_occupation_ranking.png`). The only net-positive occupations are frontline:
**welders, electricians, nurses**, then HVAC and plumbers (≈0). Bottom of the table: **customer service
(−0.23), admin / office (−0.20)**, finance, writers / translators, lawyers / paralegals, software
engineers (−0.11, by far the most discussed at 8.5k mentions). Two frontline exceptions are telling:
**drivers / logistics (−0.20)** — people expect autonomous vehicles — and hands-on healthcare aides.
That matches the hard data: the shift is "care + skilled trades", not "all blue collar".

**What it does *not* do (tested and reported honestly):**
* **No lead on official data.** Corr(Δ social index, future Δ Indeed / Δ CES) is ≈0 at every lag (−3…+6m).
* **No trading edge.** Social index rank-IC vs next-1m hedged spread: −0.02 in the AI era (3m: +0.06).
  It is therefore **not** added to the model — like the FSI, it confirms the thesis but doesn't time it.
* **AI launches:** knowledge outlook worsened in the 14 days after 12 of 20 launches, but the average
  change (−0.02) is within noise.

**Measurement quality.** Hand-labelled validation (`outputs/social_validation_round2.csv`, 60 sentences
labelled blind *after* the scorer was frozen): when the scorer assigns a stance it is right **62%** of
the time and flips the sign **18%**. It is noisy per comment but biased similarly for both groups — on
that sample it estimates the frontline-minus-knowledge gap at 0.33 vs 0.43 by hand — so aggregates are
usable and, if anything, understate the gap. Other caveats: Reddit's pre-ChatGPT "AI" sample is thin
(384 mentions, 984 career-sub comments); HN skews towards software people; r/Trades returned no
archive matches and was dropped.

## Part 2 — The model

```
                       ┌─────────────────────────── Frontline Shift Index (FSI) ─────────────────────┐
 Indeed postings ──┐   │ z(13w Δ postings spread, +10d lag) + z(6m Δ CES frontline share, +40d lag)  │
 BLS payrolls ─────┼──▶│ + z(6m Δ young-grad penalty, +40d lag); expanding-window z ⇒ no look-ahead   │──▶ kill-switch
 CPS unemployment ─┘   └──────────────────────────────────────────────────────────────────────────────┘     (½ size if FSI<−1 for 2m)
                                                                                                                   │
 16 Frontline Enablers (long) ─┐   each leg β-hedged to SPY        ┌─────────────┐                                 ▼
 17 Knowledge-Labour Sellers ──┴─▶ (252d β, lagged) ─▶ spread ──▶  │ 10% vol tgt │ ─▶ core 1x ─┬─▶ PRODUCTION book
                                                                    └─────────────┘             │
 Frontier-AI release calendar (20 events, + CLI to add new ones) ─▶ +0.5x for days +1…+10 ──────┘
```

Baskets were chosen by **business model** (revenue tied to frontline headcount vs to billed
knowledge-labour hours), and deliberately keep losers on both sides (TBI, AMN long; ACN, IT short).
ATGE and ASGN returned no Yahoo data and are excluded.

### Backtest (net of 10bp costs + monthly rebalancing, 10% vol target)

| Book | Pre-AI (2016-07 → 2022-11) Sharpe | AI era (2022-12 → 2026-11) Sharpe | AI-era CAGR | AI-era MaxDD |
|---|---|---|---|---|
| A — Structural pair | **0.00** | **1.82** | 20.4% | −9.3% |
| B — FSI tilt (0-2x) | 0.27 | 1.33 | 15.0% | −11.2% |
| B — FSI timing (±1x) | 0.51 | **−1.43** | −14.5% | −46.7% |
| C — AI-event drift only | – | 0.90 | 3.8% | −5.7% |
| **PRODUCTION (A + C, FSI kill-switch)** | 0.00 | **1.83** | **22.6%** | −9.3% |

Raw legs since ChatGPT: frontline basket **+22.6%/yr**, knowledge basket **−12.9%/yr**, SPY +18.0%/yr.
Hedged spread: −3% cumulative pre-AI, **+279% cumulative in the AI era**.

### What the research says — including what failed
* **The regime break is the edge.** Same baskets, same rules: zero Sharpe before ChatGPT, 1.8 after.
* **Robust.** AI-era alpha **39%/yr, t = 3.4** after controlling for market, industrials-minus-tech,
  size, momentum, value and quality. 500 random half-baskets: **100% positive**, median Sharpe 1.23
  (5th pct 0.91). Dropping the 10 biggest winners on both sides (FIX, EME, LINC, UTI, GWW, CHGG, TTEC,
  FVRR, GLOB, COUR) still leaves Sharpe 0.67. Positive every calendar year 2022-2026.
* **Labour data confirms the thesis but does NOT time it.** FSI rank-IC vs forward 3m spread return
  is −0.26 in the AI era: the market prices the shift *before* BLS/Indeed show it. So FSI is used only
  as a thesis kill-switch (it never fired in-sample), not as a sizing signal.
* **AI-release events: positive but not conclusive.** Avg spread drift +2.7% over days +1…+10 (13/20
  positive; +5.0% over +1…+20), vs a random-date placebo **p = 0.13**. Biggest: Claude Cowork (+7.3%),
  the Feb-2026 "SaaSpocalypse" (+15.4%), Claude 3 (+5.9%), Claude 3.7/Claude Code (+5.9%). Kept as a
  half-size overlay because it's cheap and pre-registered; would need more events to promote it.
* **Big caveat — hindsight.** The baskets were assembled in 2026 by someone who knows CHGG went −98%.
  The random-subbasket and leave-out tests reduce, not remove, that bias. FIX/EME are also AI-capex
  (data-centre electricians) winners — arguably *part* of the thesis (AI's build-out is frontline work),
  but a capex bust would hit the long leg.

### Live book (as of 2026-11-23) — `python -m frontline.live`
Gross 0.68x NAV (vol-multiplier is 0.30x because spread vol spiked in November), ~2.0% per long,
~1.8% per short, 9% short SPY to neutralise beta (long-leg β 0.82 vs short-leg 0.51). FSI +0.18
(neutral-supportive), kill-switch off, no AI-event overlay active. When a new frontier model ships:

```bash
python -m frontline.live --event 2026-12-01 "New frontier model"
```

## Part 3 — Outside-the-box trade ideas (ranked by conviction)

1. **"Sell the billable hour" pair** *(the model above)* — long frontline enablers, short
   offshore IT services / CX-BPO / white-collar staffing. Cleanest expression; backtested.
2. **Trade-school vs ed-tech** — long UTI, LINC (enrolment = direct read on people *choosing* the
   frontline) vs short COUR, CHGG-type white-collar upskilling. Earnings-event trade: enrolment
   prints are the catalyst. UTI +169%, LINC +212% vs COUR −73%, CHGG −98% since ChatGPT.
3. **Uniform rental as a frontline payroll proxy** — CTAS/UNF revenue ≈ number of people in
   uniforms. Short-dated, event-driven: trade the CTAS print vs the same-month knowledge-industry
   CES print. A "headcount you can rent" index.
4. **Customer-service BPO short into each frontier-model launch** — CNXC, TTEC, TASK are the most
   directly substitutable labour in the market; matched-pair data shows customer-service postings
   now lag even food service. Pair against ROL/CTAS to strip out consumer cycle.
5. **Indian IT vs US skilled-trades contractors** — short INFY/WIT/CTSH vs long EME/FIX: AI deflates
   offshore labour arbitrage while AI data centres need US electricians. Two-sided AI exposure.
6. **Second-order: frontline wage squeeze (the un-priced leg)** — if frontline wages start to
   outrun knowledge wages (lens 5 is the trigger), *short* frontline-labour-intensive margin stories
   (casual dining, home health, hospital operators) vs *long* AI-margin beneficiaries that cut
   knowledge headcount (large insurers, banks). Not yet live — wait for the wage signal.
7. **Office vs field** — short commercial-office REITs (fewer white-collar desks) vs long
   self-storage/industrial serving trades; fewer knowledge seats is a slow-burn tenancy event.
8. **Tools & workwear** — SNA, BOOT, GWW, FAST: a tradie's spend scales with tradie headcount and
   wages; low-beta way to own the shift.

## Risks / what would kill the thesis
* **Embodied AI / humanoid robots** make frontline work automatable → watch robotics deployment data.
* **AI-capex bust** hits FIX/EME/WSO (long leg) and rallies IT services on relief.
* **Rate cuts reviving tech hiring** — the 2022 confound in reverse; FSI kill-switch is the guard.
* **Crowding:** the spread had a momentum loading of 0.94 in the AI era — the trade *is* momentum now,
  vulnerable to momentum crashes (cf. the −37% spread year in 2020).

## Repository
| Path | What |
|---|---|
| `frontline/config.py` | All classifications (occupations, industries, baskets, AI events), decided a priori |
| `frontline/data.py` | FRED + Yahoo loaders with CSV cache in `data/raw/` |
| `frontline/evidence.py` | Part 1: five labour-market lenses + DiD |
| `frontline/strategy.py` | Part 2: baskets, hedged spread, FSI, books, AI event study |
| `frontline/robustness.py` | Factor regression, leave-out, random sub-baskets |
| `frontline/live.py` | Today's target weights; register new AI events |
| `frontline/social_collect.py` | Reddit (Arctic Shift) + Hacker News (Algolia) collector, resumable |
| `frontline/social.py` | Occupation outlook scorer, pivot-intent index, RoBERTa cross-check, lead/lag + IC tests |
| `outputs/` | Figures, stats JSON, performance tables, live book |

Social data: `python -m frontline.social_collect` (≈2-4 h, resumable) then `python -m frontline.social`
(`--no-transformer` skips RoBERTa); raw comments are git-ignored.

Sources: Reddit via Arctic Shift, Hacker News via Algolia; BLS Current Employment Statistics & CPS, BLS JOLTS, Indeed Hiring Lab (all via FRED);
prices via Yahoo Finance. Not investment advice; research prototype.
