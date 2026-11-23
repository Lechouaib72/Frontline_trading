"""Collect public social-media discussion about AI and the future of jobs.

Sources (public research APIs only; no login, no author handles stored):
  * Reddit — via the Arctic Shift archive (https://arctic-shift.photon-reddit.com)
  * Hacker News — via the Algolia HN Search API

Sampling: for each subreddit, the first 7 days of every month, up to 100 comments that mention "AI" (Reddit), and up to 1000 per query-month on HN. Counts are
therefore *samples*: all downstream metrics are ratios within the sample, never
raw volumes.

    python -m frontline.social_collect            # incremental; skips finished chunks
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import requests

from . import config as C

SOC = C.RAW / "social"
UA = {"User-Agent": "frontline-research/0.1 (academic labour-market study)"}

SUBREDDITS = {
    # careers / people deciding what to do
    "careerguidance": "career", "findapath": "career",
    # knowledge workers
    "cscareerquestions": "knowledge",
    # frontline workers
    "Trades": "frontline", "electricians": "frontline", "nursing": "frontline",
    # general AI / future-of-work discussion
    "Futurology": "ai",
}
HN_QUERIES = ["AI jobs", "trade school", "blue collar", "learn a trade", "electrician", "plumber",
              "AI replace", "junior developer"]
START, END = "2021-01-01", "2026-11-23"


def _windows():
    """First 7 days of each month: the archive's full-text search times out on wider
    windows, and a fixed within-month slice keeps months comparable."""
    for m in pd.date_range(START, END, freq="MS"):
        yield m, min(m + pd.Timedelta(days=7), pd.Timestamp(END))


def _get(url, params, tries=5):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=90)
            if r.status_code in (422, 429, 503) or (r.ok and r.json().get("error")):
                time.sleep(30 * (i + 1)); continue           # server asked us to slow down
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == tries - 1:
                return None
            time.sleep(2 ** i)


def reddit_chunk(sub, lo, hi):
    d = _get("https://arctic-shift.photon-reddit.com/api/comments/search",
             {"subreddit": sub, "body": "AI", "after": lo.strftime("%Y-%m-%d"),
              "before": hi.strftime("%Y-%m-%d"), "limit": 100,
              "fields": "id,created_utc,body,subreddit,score"})
    time.sleep(1.5)                                          # politeness gap, sequential client
    if d is None or d.get("data") is None:
        return None
    return [{"src": "reddit", "id": x["id"], "ts": x["created_utc"], "community": x["subreddit"],
             "score": x.get("score"), "text": x["body"]} for x in d["data"]]


def hn_chunk(q, lo, hi):
    d = _get("https://hn.algolia.com/api/v1/search",
             {"query": q, "tags": "comment", "hitsPerPage": 1000,
              "numericFilters": f"created_at_i>{int(lo.timestamp())},created_at_i<{int(hi.timestamp())}"})
    if d is None:
        return None
    return [{"src": "hn", "id": h["objectID"], "ts": h["created_at_i"], "community": f"hn:{q}",
             "score": h.get("points"), "text": h.get("comment_text") or ""} for h in d["hits"]]


def collect():
    SOC.mkdir(parents=True, exist_ok=True)
    done_path = SOC / "_done.txt"
    done = set(done_path.read_text().split("\n")) if done_path.exists() else set()
    jobs = []
    for lo, hi in _windows():
        for sub in SUBREDDITS:
            jobs.append((f"r7|{sub}|{lo.date()}", reddit_chunk, (sub, lo, hi)))
    for m in pd.date_range(START, END, freq="MS"):
        hi = min(m + pd.offsets.MonthBegin(1), pd.Timestamp(END))
        for q in HN_QUERIES:
            jobs.append((f"h|{q}|{m.date()}", hn_chunk, (q, m, hi)))
    jobs = [j for j in jobs if j[0] not in done]
    print(f"{len(jobs)} chunks to fetch")
    out = open(SOC / "posts.jsonl", "a")
    rj = [j for j in jobs if j[0].startswith("r7|")]
    hj = [j for j in jobs if j[0].startswith("h|")]
    with ThreadPoolExecutor(1) as rex, ThreadPoolExecutor(2) as hex_, open(done_path, "a") as dl:
        futs = {rex.submit(f, *a): key for key, f, a in rj}
        futs.update({hex_.submit(f, *a): key for key, f, a in hj})
        for n, fu in enumerate(as_completed(futs), 1):
            rows = fu.result()
            if rows is None:
                print("FAILED", futs[fu], flush=True)
                continue
            for r in rows:
                out.write(json.dumps(r) + "\n")
            out.flush()
            dl.write(futs[fu] + "\n"); dl.flush()
            if n % 50 == 0:
                print(n, datetime.now(timezone.utc).strftime("%H:%M:%S"), flush=True)
    out.close()
    print("DONE", flush=True)


def load():
    rows = []
    with open(SOC / "posts.jsonl") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:      # tolerate a half-written last line
                pass
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(["src", "id"])
    df["date"] = pd.to_datetime(df["ts"], unit="s")
    # keep only the fixed first-7-days slice for Reddit (drops rows from earlier pilot runs)
    df = df[(df.src != "reddit") | (df.date.dt.day <= 7)]
    return df.reset_index(drop=True)


if __name__ == "__main__":
    collect()
