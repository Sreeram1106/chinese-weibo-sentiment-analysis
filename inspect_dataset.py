"""Dataset audit: size, duplicates, label balance, COVID keyword screen."""
from __future__ import annotations

import json

from .config import COVID_TERMS, RESULTS_DIR
from .data import clean_dataset, load_raw


def main():
    RESULTS_DIR.mkdir(exist_ok=True)          # original crashed when results/ was missing
    raw = load_raw()
    clean, audit = clean_dataset(raw)
    print("Raw rows:", len(raw))
    print("Exact duplicate texts in raw file:", int(raw["text"].duplicated().sum()))
    print("Audit:", json.dumps(audit))
    print("\nLabel counts (raw -> deduplicated):")
    rc, cc = raw["label"].value_counts().sort_index(), clean["label"].value_counts().sort_index()
    for c in rc.index:
        print(f"  {c}: {rc[c]:>5} -> {cc.get(c, 0):>5}")
    mask = clean["text"].str.contains("|".join(COVID_TERMS), case=False, regex=True)
    print(f"\nRows matching a COVID keyword: {int(mask.sum())} / {len(clean)} ({mask.mean():.1%}) "
          "- keyword screen only, NOT a validated COVID subset; the file has no date column, so no time-trend analysis is possible.")
    clean.loc[mask, ["text", "label"]].head(200).to_csv(RESULTS_DIR / "covid_keyword_sample.csv", index=False, encoding="utf-8-sig")
    (RESULTS_DIR / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
