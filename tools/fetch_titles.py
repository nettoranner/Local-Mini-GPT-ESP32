"""Most viewed English Wikipedia articles -> titles.txt (most viewed first).

Uses the public Wikimedia pageviews API (top 1000 articles per month), sums the
views over the chosen months and writes one title per line. Feed the result to
build_qa.py --titles titles.txt --n 500

  python fetch_titles.py --year 2023 --months 1-10 --top 3000 --out titles.txt
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

API = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/top/"
       "en.wikipedia/all-access/{y}/{m:02d}/all-days")
HEADERS = {"User-Agent": "build_qa-titles/1.0 (personal learning project)"}

# service pages that show up in the top list
SKIP = re.compile(r"^(Main_Page|Special:|Wikipedia:|Portal:|Help:|File:|Category:|Template:|"
                  r"Talk:|User:|Draft:|-$)|^\d{4}$|^Deaths_in_|^List_of_|_\(disambiguation\)$")


def fetch_month(y, m):
    req = urllib.request.Request(API.format(y=y, m=m), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    return data["items"][0]["articles"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--months", default="1-10", help="e.g. 1-10 or 3-6")
    ap.add_argument("--top", type=int, default=3000)
    ap.add_argument("--out", default="titles.txt")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.months.split("-"))

    views = {}
    for m in range(lo, hi + 1):
        try:
            arts = fetch_month(a.year, m)
        except (urllib.error.URLError, KeyError, ValueError) as e:
            print(f"  {a.year}-{m:02d}: skipped ({e})", file=sys.stderr)
            continue
        for it in arts:
            t = it["article"]
            if not SKIP.search(t):
                views[t] = views.get(t, 0) + it["views"]
        print(f"  {a.year}-{m:02d}: {len(arts)} articles", file=sys.stderr)
        time.sleep(0.5)

    if not views:
        sys.exit("no month could be fetched; check internet access / --year / --months")
    top = sorted(views, key=views.get, reverse=True)[:a.top]
    with open(a.out, "w", encoding="utf-8") as f:
        for t in top:
            f.write(t.replace("_", " ") + "\n")
    print(f"wrote {len(top)} titles to {a.out}; first: {', '.join(t.replace('_',' ') for t in top[:5])}")


if __name__ == "__main__":
    main()
