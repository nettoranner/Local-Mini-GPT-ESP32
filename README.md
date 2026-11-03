# Local-Mini-GPT-ESP32
Trained model from scratch and runnig fully offline on ESP32-S3 N16R8

# Download and format data
1. Dowload dataset

```bash
python tools/build_qa.py 

```

2. Format data

```bash
python tools/build_qa.py --parquet-dir tools/202311101.en --titles training/titles.txt --n 3000 --out training/data/qa3000

```

## Data sources and license

The training answers are derived from the English Wikipedia
(dataset `wikimedia/wikipedia`, snapshot `20231101.en`, on Hugging Face).
Wikipedia text is available under the
[Creative Commons Attribution-ShareAlike 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/)
(CC BY-SA 4.0). Authors of the articles are listed in each article's history on
https://en.wikipedia.org.

Changes made: only the first one or two sentences of each article are used,
text in parentheses is removed, and characters are converted to plain ASCII.

The list of article titles was built from the Wikimedia pageviews API (2023).

Code in this repository is released under the MIT License.
Model weights trained on Wikipedia-derived text are released under CC BY-SA 4.0.
