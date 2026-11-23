"""Data loaders with an on-disk CSV cache (data/raw/).  Pass refresh=True to re-download."""
import io
import time

import pandas as pd
import requests

from . import config as C

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def _fred_one(sid: str, refresh: bool = False) -> pd.Series:
    path = C.RAW / f"fred_{sid}.csv"
    if refresh or not path.exists():
        for attempt in range(4):
            try:
                r = requests.get(FRED_URL.format(sid), timeout=30)
                r.raise_for_status()
                if r.text.lstrip().startswith("<"):
                    raise ValueError(f"FRED returned HTML for {sid}")
                path.write_text(r.text)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    df = pd.read_csv(path)
    df.columns = ["date", sid]
    df["date"] = pd.to_datetime(df["date"])
    return pd.to_numeric(df.set_index("date")[sid], errors="coerce")


def fred(ids, refresh: bool = False) -> pd.DataFrame:
    return pd.concat([_fred_one(s, refresh) for s in ids], axis=1)


def indeed(refresh: bool = False):
    ids = {f"IHLIDXUSTP{k}": k for k in {**C.INDEED_FRONTLINE, **C.INDEED_KNOWLEDGE}}
    df = fred(list(ids), refresh).rename(columns=ids)
    return df.dropna(how="all")


def prices(refresh: bool = False) -> pd.DataFrame:
    path = C.RAW / "prices.csv"
    tickers = sorted({*C.FRONTLINE_LONG, *C.KNOWLEDGE_SHORT, C.BENCH})
    if refresh or not path.exists():
        import yfinance as yf
        px = yf.download(tickers, start="2014-01-01", auto_adjust=True, progress=False)["Close"]
        px.to_csv(path)
    px = pd.read_csv(path, index_col=0, parse_dates=True)
    return px[[t for t in tickers if t in px.columns]]
