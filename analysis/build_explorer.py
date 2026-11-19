"""Embed results_v2/report/explorer_data.json into the explorer page.

Usage: python analysis/build_explorer.py [--data results_v2/report/explorer_data.json]
                                         [--out results_v2/report/explorer.html]
                                         [--pending modelset_abstract="re-running"]
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results_v2/report/explorer_data.json")
    ap.add_argument("--out", default="results_v2/report/explorer.html")
    ap.add_argument("--pending", nargs="*", default=[], help="key=label for datasets without results yet")
    args = ap.parse_args()
    data = Path(args.data).read_text()
    pending = dict(p.split("=", 1) for p in args.pending)
    page = (HERE / "explorer_template.html").read_text()
    page = page.replace("/*__DATA__*/null", data).replace("/*__PENDING__*/{}", json.dumps(pending))
    Path(args.out).write_text(page)
    print(f"wrote {args.out} ({len(page) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
