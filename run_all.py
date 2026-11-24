"""Reproduce the full study: evidence -> strategy backtests -> robustness -> social sentiment -> live book."""
import json
import sys
import warnings

warnings.filterwarnings("ignore")

from frontline import evidence, strategy, robustness, live  # noqa: E402
from frontline import social_collect as SC  # noqa: E402

if __name__ == "__main__":
    if "--refresh" in sys.argv:
        from frontline import data as D, config as C
        D.indeed(refresh=True)
        D.fred([*C.CES_FRONTLINE, *C.CES_KNOWLEDGE, *C.AHE_FRONTLINE, *C.AHE_KNOWLEDGE, *C.OTHER_FRED], refresh=True)
        D.prices(refresh=True)
    ev, _ = evidence.run()
    st, perf, _ = strategy.run()
    rb = robustness.run()
    if (SC.SOC / "posts.jsonl").exists():          # social layer needs `python -m frontline.social_collect` first
        from frontline import social
        social.run(use_transformer="--no-transformer" not in sys.argv)
    book, meta = live.target_book()
    book.to_csv("outputs/live_target_book.csv")
    print(json.dumps({"evidence": ev, "live": meta}, indent=2, default=str))
    print(perf.to_string())
