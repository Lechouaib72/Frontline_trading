"""Reproduce the full study: evidence -> strategy backtests -> robustness -> live book."""
import json
import sys
import warnings

warnings.filterwarnings("ignore")

from frontline import evidence, strategy, robustness, live  # noqa: E402

if __name__ == "__main__":
    if "--refresh" in sys.argv:
        from frontline import data as D, config as C
        D.indeed(refresh=True)
        D.fred([*C.CES_FRONTLINE, *C.CES_KNOWLEDGE, *C.AHE_FRONTLINE, *C.AHE_KNOWLEDGE, *C.OTHER_FRED], refresh=True)
        D.prices(refresh=True)
    ev, _ = evidence.run()
    st, perf, _ = strategy.run()
    rb = robustness.run()
    book, meta = live.target_book()
    book.to_csv("outputs/live_target_book.csv")
    print(json.dumps({"evidence": ev, "live": meta}, indent=2, default=str))
    print(perf.to_string())
