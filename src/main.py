"""CLI: python -m src.main data/sample_reviews.csv [--dry-run]"""
import argparse
import csv
import os
import time
from pathlib import Path

from src.classifier import classify_dry_run, classify_review

try:
    from dotenv import load_dotenv
except ImportError:  # optional
    def load_dotenv():
        pass


def main():
    parser = argparse.ArgumentParser(description="Auto-classify reviews with AI")
    parser.add_argument("input_csv")
    parser.add_argument("--column", default="review", help="column with review text")
    parser.add_argument("--dry-run", action="store_true", help="no API calls")
    args = parser.parse_args()

    load_dotenv()
    with open(args.input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows_in = list(reader)
    if args.column not in fields:
        raise SystemExit(f"Column '{args.column}' not found. Columns: {fields}")

    client = None
    if not args.dry_run:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("Set GEMINI_API_KEY in .env (or use --dry-run)")
        client = genai.Client(api_key=api_key)

    new_cols = ["sentiment", "topic", "urgency", "summary"]
    out_rows = []
    for i, row in enumerate(rows_in, 1):
        text = row.get(args.column) or ""
        if args.dry_run:
            result = classify_dry_run(text)
        else:
            try:
                result = classify_review(client, text)
            except Exception as e:  # keep going if one request fails
                print(f"Row {i} failed: {e}")
                result = {"sentiment": "", "topic": "", "urgency": "", "summary": "ERROR"}
            time.sleep(float(os.getenv("REQUEST_DELAY", "1")))  # stay under free-tier limits
        out_rows.append({**row, **result})
        print(f"[{i}/{len(rows_in)}] done")

    out_path = Path(args.input_csv).with_name(Path(args.input_csv).stem + "_classified.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields + new_cols)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
