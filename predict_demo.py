"""Command-line demo.   python -m src.predict_demo [--text "..."] [--file posts.txt]"""
from __future__ import annotations

import argparse

from .config import LABEL_NAMES_ARE_VERIFIED, MODEL_PATH
from .predict import predict


def show(res: dict):
    print(f"\n  {res['text'][:80]}")
    for r in res["top"]:
        bar = "#" * round(r["probability"] * 30)
        print(f"   {r['label']} {r['name']:<16} {r['probability']:6.1%}  {bar}")


def main():
    ap = argparse.ArgumentParser(description="Chinese Weibo 7-class emotion classifier")
    ap.add_argument("--text", help="classify one post")
    ap.add_argument("--file", help="UTF-8 text file, one post per line")
    a = ap.parse_args()
    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. First run: python -m src.train")
    if not LABEL_NAMES_ARE_VERIFIED:
        print("(class names are tentative, inferred from top words; IDs 0-6 are the source labels)")
    if a.text:
        return show(predict(a.text))
    if a.file:
        for line in open(a.file, encoding="utf-8"):
            if line.strip():
                show(predict(line.strip()))
        return
    print("Type a Chinese post and press Enter (empty line to quit).")
    while True:
        try:
            t = input("\ntext> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not t:
            break
        show(predict(t))


if __name__ == "__main__":
    main()
