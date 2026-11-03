"""Build question->answer pairs for the ESP32 char-model from wikimedia/wikipedia.

Answer  = the first 1..k sentences of the article lead (<= --max-chars chars),
          parentheses removed, text folded to the 97-char vocabulary.
Question = several templates ("who is X?", "tell me about X", ...).

Selection of articles:
  --titles FILE   one article title per line, most important first
                  (see fetch_titles.py); --n takes the first n of them
  otherwise       the --n longest articles that pass the filters (length is a
                  rough proxy for importance; the dataset has no page views)

Source of articles:
  --parquet-dir D   local *.parquet shards (columns: title, text)
  --hf              stream wikimedia/wikipedia 20231101.en via `datasets`
                    (iterates over the whole dataset, ~20 GB of traffic)

Output (in --out):
  train.jsonl     {"q","a"} with --paraphrases questions per article
  val.jsonl       {"q","a"} with two question templates NOT used for training
  articles.jsonl  {"title","person","a"} one line per article
  vocab.json      97 symbols: ASCII 32..126, "\\n", END (\\x03)

Typical run:
  python build_qa.py --parquet-dir ./wiki_en --n 1000 --out data1000
"""
import argparse
import glob
import heapq
import itertools
import json
import os
import random
import re
import sys
import unicodedata

END = "\x03"
VOCAB = ["\x00"] + [chr(c) for c in range(32, 127)] + ["\n", END]  # 95 + 1 + 1 = 97
VOCAB_SET = set(VOCAB[1:])

_MAP = {"–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-",
        "‘": "'", "’": "'", "“": '"', "”": '"',
        " ": " ", "…": "...", "•": "-",
        "\u0153": "oe", "\u0152": "Oe", "\u00e6": "ae", "\u00c6": "Ae",
        "\u00f8": "o", "\u00d8": "O", "\u00df": "ss", "\u0142": "l", "\u0141": "L",
        "\u0111": "d", "\u0110": "D", "\u00f0": "d", "\u00fe": "th"}

MONTHS = ("January|February|March|April|May|June|July|August|September|"
          "October|November|December")
PERSON_RE = re.compile(
    r"\([^)]*\bborn\b[^)]*\)"
    r"|\([^)]*\b(?:%s)\b[^)]*\b\d{3,4}\b[^)]*[–—-][^)]*\b\d{3,4}\b[^)]*\)" % MONTHS)

# titles that have a date range in the lead but are events, not people
EVENT_RE = re.compile(
    r"\b(War|Wars|Battle|Siege|Treaty|Revolution|Crisis|Attack|Attacks|Massacre|Conflict|"
    r"Operation|Campaign|Expedition|Election|Elections|Conference|Games|Championship|"
    r"Season|Festival|Rebellion|Uprising|Coup|Strike|Riots|Dynasty|Empire|Era|Age|Period|"
    r"Famine|Pandemic|Epidemic|Disaster|Crash|Congress|Summit|Cup|Olympics)\b")

PERSON_T = ["who is {t}?", "who was {t}?", "who is {t}", "who was {t}",
            "tell me about {t}", "tell me about {t}.", "{t}", "{t}?",
            "what do you know about {t}?", "{t} biography", "explain who {t} is"]
OTHER_T = ["what is {t}?", "what is {t}", "tell me about {t}", "tell me about {t}.",
           "{t}", "{t}?", "explain {t}", "describe {t}", "define {t}",
           "what do you know about {t}?", "{t} explained"]
HELD_OUT_T = ["what can you tell me about {t}?", "give me info on {t}"]

BAD_PREFIX = ("List of", "Lists of", "Index of", "Outline of", "Timeline of")

# service / year-based / catalogue pages nobody asks about
BAD_TITLE_RE = re.compile(
    r"^(\d{3,4}\b|Deaths in|Glossary of|Bibliography of|Chronology of|Characters of|"
    r"Responses to|Impact of)|\bHonours\b|\b(discography|filmography|bibliography)$|"
    r"\bin (sports|film|music|literature|television|science|politics)\b")

# adult / offensive titles: not for a public demo
BLOCK_RE = re.compile(r"porn|xxx|xvideos|xnxx|xhamster|onlyfans|\bsex\w*|mia khalifa|"
                      r"2 girls 1 cup|gayniggers|nigger|rule 34", re.I)

# person check runs on the FIRST top-level parenthetical of the lead (handles nested "(listen)")
PERSON_PAR_RE = re.compile(
    r"\bborn\b|\b(?:%s)\b.*\b\d{3,4}\b.*[\u2013\u2014-].*\b\d{3,4}\b|\bBC\b.*[\u2013\u2014-]|[\u2013\u2014-].*\bBC\b"
    % MONTHS)


def first_parenthetical(s):
    depth, start = 0, 0
    for i, ch in enumerate(s):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
            if depth == 0:
                return s[start + 1:i]
    return ""


# person detection from the cleaned answer (used when the lead has no usable date parenthetical)
NEG_RE = re.compile(r"\b(film|films|movie|series|television|sitcom|miniseries|album|song|single|tour|"
                    r"company|corporation|club|team|band|group|organi[sz]ation|website|app|software|"
                    r"video game|game|franchise|country|city|river|mausoleum|liner|ship|festival|league|"
                    r"cup|award|awards|conflict|war|battle|attack|election|event|brand|service|platform|"
                    r"network|sport|stadium|building|monument|bank|airline|aircraft|operating system|"
                    r"day|holiday|feast|celebrated|religious|language model)\b", re.I)

OCC = (r"actor|actress|singer|rapper|musician|politician|footballer|player|cricketer|boxer|wrestler|"
       r"fighter|writer|author|poet|novelist|businessman|businesswoman|entrepreneur|scientist|physicist|"
       r"mathematician|philosopher|queen|king|emperor|president|prime minister|comedian|guitarist|"
       r"drummer|pianist|composer|director|producer|journalist|presenter|athlete|driver|golfer|"
       r"swimmer|general|pope|prophet|criminal|fraudster|outlaw|statesman|monarch|duke|duchess|prince|"
       r"princess|engineer|inventor|artist|painter|activist|lawyer|judge|astronaut|economist|chef|dancer|"
       r"drug lord|serial killer|mobster|gangster|preacher|philanthropist|model")
PERSON_TEXT_RE = re.compile(r"\b(?:was|is|were)\s+(?:an?\s+|the\s+)?(?:[\w'-]+,?\s+){0,3}?(?:%s)\b" % OCC, re.I)
PRON_RE = re.compile(r"\b(he|she|his|her|him)\b", re.I)


def person_from_text(ans):
    first = re.split(r"(?<=[.!?])\s+", ans)[0]
    if NEG_RE.search(first):
        return False
    return bool(PERSON_TEXT_RE.search(first) or PRON_RE.search(ans))


# abbreviations / initials that end with a dot but do not end a sentence
ABBR_END = re.compile(r"(?:\b[A-Z]|U\.S|U\.K|St|Mr|Mrs|Ms|Dr|Jr|Sr|Mt|Gen|Col|Lt|Capt|"
                      r"Sgt|Prof|Rev|Hon|vs|No|Inc|Ltd|Co|Ft|Fr)\.$")

# grammar leftovers after removing parentheses: "called the, was"
DANGLING_RE = re.compile(r"\b(?:the|a|an|of|and|or|by|to|in|called|named|as)\s*[,;.]")


def strip_parens(s):
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\([^()]*\)", "", s)
    s = re.sub(r"\[[^\]]*\]", "", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def to_ascii(s):
    for k, v in _MAP.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def fold(s, max_dropped=2):
    """ASCII-fold and drop symbols outside the vocabulary; None if too lossy."""
    s = to_ascii(s)
    dropped = sum(1 for c in s if c not in VOCAB_SET)
    if dropped > max_dropped:
        return None
    return "".join(c for c in s if c in VOCAB_SET)


def make_answer(first_para, max_chars, min_chars):
    clean = fold(strip_parens(first_para))
    if not clean:
        return None
    clean = re.sub(r"\s{2,}", " ", clean).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", clean)
    sents = []
    for part in parts:
        if sents and ABBR_END.search(sents[-1]):
            sents[-1] += " " + part
        else:
            sents.append(part)
    acc = ""
    for s in sents:
        cand = (acc + " " + s).strip()
        if len(cand) > max_chars:
            break
        acc = cand
    if not (acc[:1].isupper() or acc[:1].isdigit()):
        return None
    if len(acc) < min_chars or acc[-1] not in ".!?" or ABBR_END.search(acc):
        return None
    if DANGLING_RE.search(acc):
        return None
    return acc


def make_title(title):
    t = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
    t = fold(t, max_dropped=0)
    if not t or len(t) > 60:
        return None
    return t


def process(title, text, max_chars, min_chars):
    if title.startswith(BAD_PREFIX) or "(disambiguation)" in title or BAD_TITLE_RE.search(title) or BLOCK_RE.search(title):
        return None
    first = text.split("\n", 1)[0].strip()
    if len(first) < 60 or "may refer to" in first or "may also refer to" in first:
        return None
    ans = make_answer(first, max_chars, min_chars)
    t = make_title(title)
    if not ans or not t:
        return None
    person = (bool(PERSON_PAR_RE.search(first_parenthetical(first))) or person_from_text(ans)) \
        and not EVENT_RE.search(t) and not re.match(r"\d", t)
    return {"title": t, "person": person, "a": ans}


def iter_arrow(d):
    """Dataset saved with datasets.save_to_disk (dataset_dict.json + train/*.arrow)."""
    from datasets import load_from_disk
    ds = load_from_disk(d)
    if hasattr(ds, "keys"):  # DatasetDict
        ds = ds["train"]
    ds = ds.select_columns(["title", "text"])
    for batch in ds.iter(batch_size=2000):
        for t, x in zip(batch["title"], batch["text"]):
            yield t, x


def iter_parquet(d):
    if os.path.exists(os.path.join(d, "dataset_dict.json")) or \
       os.path.exists(os.path.join(d, "state.json")):
        yield from iter_arrow(d)
        return
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True))
    if not files:
        sys.exit(f"No .parquet files and no dataset_dict.json found in {d}")
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["title", "text"], batch_size=2000):
            for t, x in zip(batch.column("title").to_pylist(), batch.column("text").to_pylist()):
                yield t, x


def iter_hf():
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    for row in ds:
        yield row["title"], row["text"]


def questions(item, templates, rng, k):
    t = item["title"]
    pool = [x.format(t=t) for x in templates]
    out = []
    for q in pool:
        if rng.random() < 0.25:
            q = q.lower()
        out.append(q)
    rng.shuffle(out)
    return list(dict.fromkeys(out))[:k]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--parquet-dir")
    src.add_argument("--hf", action="store_true")
    ap.add_argument("--titles", help="file with one title per line")
    ap.add_argument("--n", type=int, default=500, help="number of articles")
    ap.add_argument("--max-chars", type=int, default=300, help="max answer length")
    ap.add_argument("--min-chars", type=int, default=80)
    ap.add_argument("--paraphrases", type=int, default=12, help="questions per article in train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    rows = iter_parquet(a.parquet_dir) if a.parquet_dir else iter_hf()

    wanted, order = None, []
    if a.titles:  # file order = priority (e.g. most viewed first); --n takes the first n found
        order = [l.strip() for l in open(a.titles, encoding="utf-8") if l.strip()]
        wanted = set(order)
    found = {}

    heap, counter, seen = [], itertools.count(), 0
    picked = []
    for title, text in rows:
        seen += 1
        if wanted is not None and title not in wanted:
            continue
        item = process(title, text, a.max_chars, a.min_chars)
        if item is None:
            continue
        if wanted is not None:
            found.setdefault(title, item)
        else:
            key = (len(text), next(counter))
            if len(heap) < a.n:
                heapq.heappush(heap, (key, item))
            elif key > heap[0][0]:
                heapq.heapreplace(heap, (key, item))
        if seen % 500000 == 0:
            print(f"  scanned {seen} articles...", file=sys.stderr)

    if wanted is None:
        picked = [it for _, it in sorted(heap, key=lambda x: x[0], reverse=True)]
    else:
        picked = [found[t] for t in order if t in found][:a.n]

    # one answer per cleaned title
    uniq = {}
    for it in picked:
        uniq.setdefault(it["title"].lower(), it)
    items = list(uniq.values())

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "train.jsonl"), "w", encoding="utf-8") as ft, \
         open(os.path.join(a.out, "val.jsonl"), "w", encoding="utf-8") as fv, \
         open(os.path.join(a.out, "articles.jsonl"), "w", encoding="utf-8") as fa:
        for it in items:
            tpl = PERSON_T if it["person"] else OTHER_T
            for q in questions(it, tpl, rng, a.paraphrases):
                ft.write(json.dumps({"q": q, "a": it["a"]}, ensure_ascii=False) + "\n")
            for q in questions(it, HELD_OUT_T, rng, 2):
                fv.write(json.dumps({"q": q, "a": it["a"]}, ensure_ascii=False) + "\n")
            fa.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(VOCAB, f, ensure_ascii=False)

    lens = [len(i["a"]) for i in items]
    print(f"scanned {seen}, kept {len(items)} articles "
          f"({sum(i['person'] for i in items)} persons), "
          f"answer length avg {sum(lens)/max(1,len(lens)):.0f}, max {max(lens, default=0)}; "
          f"vocab {len(VOCAB)} symbols -> {a.out}")


if __name__ == "__main__":
    main()
