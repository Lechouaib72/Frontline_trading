"""Live book: today's target weights per ticker (fraction of NAV).

    python -m frontline.live                         # uses cached data
    python -m frontline.live --refresh               # re-download prices + FRED first
    python -m frontline.live --event 2026-11-20 "New frontier model"   # register an AI release

New AI events are appended to data/ai_events_extra.csv and switch on the
+0.5x event overlay for the following 10 trading days.
"""
import argparse
import json

import pandas as pd

from . import config as C
from . import data as D
from .strategy import (basket_returns, hedged_spread, vol_scale, frontline_shift_index,
                       kill_switch)

EXTRA = C.ROOT / "data" / "ai_events_extra.csv"


def all_events():
    ev = dict(C.AI_EVENTS)
    if EXTRA.exists():
        ev.update(pd.read_csv(EXTRA, dtype=str).set_index("date")["event"].to_dict())
    return ev


def target_book(refresh=False):
    px = D.prices(refresh)
    if refresh:
        D.indeed(refresh=True)
    b, _ = basket_returns(px)
    spread, bf, bk = hedged_spread(b)
    vs = vol_scale(spread)
    fsi, comps = frontline_shift_index(spread.index)
    idx = spread.index
    ev_pos = pd.Series(0.0, index=idx)
    live_events = []
    for d, name in all_events().items():
        p = idx.searchsorted(pd.Timestamp(d))
        if p < len(idx):
            ev_pos.iloc[p: p + 10] = 1.0
            if p >= len(idx) - 10:
                live_events.append(f"{d} {name}")
    ks = kill_switch(fsi)
    t = idx[-1]
    # Position is sized with information through today's close and held tomorrow.
    vol_mult = float(vs.reindex(idx).ffill().iloc[-1])
    gross_mult = vol_mult * (float(ks.iloc[-1]) + 0.5 * float(ev_pos.iloc[-1]))
    last_px = px.ffill().iloc[-1]
    fl = [k for k in C.FRONTLINE_LONG if k in px and pd.notna(last_px[k])]
    kn = [k for k in C.KNOWLEDGE_SHORT if k in px and pd.notna(last_px[k])]
    w = {k: gross_mult / len(fl) for k in fl}
    w.update({k: -gross_mult / len(kn) for k in kn})
    w[C.BENCH] = -gross_mult * (float(bf.iloc[-1]) - float(bk.iloc[-1]))
    book = pd.DataFrame({"weight": w}).rename_axis("ticker")
    book["side"] = book.weight.map(lambda x: "LONG" if x > 0 else "SHORT")
    book["name"] = [C.FRONTLINE_LONG.get(k) or C.KNOWLEDGE_SHORT.get(k) or "beta hedge" for k in book.index]
    meta = {"as_of": str(t.date()), "vol_multiplier": round(vol_mult, 2),
            "kill_switch": round(float(ks.iloc[-1]), 2), "event_overlay_on": bool(ev_pos.iloc[-1] > 0),
            "live_events": live_events, "gross_exposure": round(book.weight.abs().sum(), 2),
            "net_exposure": round(book.weight.sum(), 2),
            "FSI": round(float(fsi.iloc[-1]), 2), "FSI_components": {k: round(float(v), 2) for k, v in comps.ffill().iloc[-1].items()},
            "beta_frontline_leg": round(float(bf.iloc[-1]), 2), "beta_knowledge_leg": round(float(bk.iloc[-1]), 2)}
    return book.sort_values("weight"), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--event", nargs=2, metavar=("DATE", "NAME"))
    a = ap.parse_args()
    if a.event:
        row = pd.DataFrame([{"date": a.event[0], "event": a.event[1]}])
        row.to_csv(EXTRA, mode="a", header=not EXTRA.exists(), index=False)
    book, meta = target_book(a.refresh)
    C.OUT.mkdir(exist_ok=True)
    book.to_csv(C.OUT / "live_target_book.csv")
    (C.OUT / "live_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(book.round(4).to_string())


if __name__ == "__main__":
    main()
