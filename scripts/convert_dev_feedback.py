"""
Convert developer overrides (/train) and ratings (/rate) into dataset samples that
can be consumed by the training pipeline.

Usage as script:
  python scripts/convert_dev_feedback.py --output data/training/dev_feedback.jsonl

Usage from Python:
  from scripts.convert_dev_feedback import merge_feedback_into_dataset
  merge_feedback_into_dataset(Path("data/training/kitsu_personality.jsonl"), Path("data/training/merged.jsonl"))

By default this will convert overrides and optionally ratings, and append them to
an existing dataset or create a new one.
"""

from pathlib import Path
import json
from typing import Optional

OVERRIDES_PATH = Path("logs/response_overrides.jsonl")
RATINGS_PATH = Path("logs/ratings.jsonl")


def _iter_overrides(path: Path = OVERRIDES_PATH):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                prompt = (obj.get("original") or "").strip()
                response = (obj.get("override") or "").strip()
                if prompt and response:
                    sample = {
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response},
                        ],
                        "metadata": {"source": "override"}
                    }
                    yield sample
            except Exception:
                continue


def _iter_ratings(path: Path = RATINGS_PATH, min_rating: Optional[int] = None):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                score = obj.get("score")
                prompt = obj.get("prompt")
                response = obj.get("response")
                if prompt and response:
                    if min_rating is not None and (score is None or int(score) < int(min_rating)):
                        continue
                    sample = {
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response},
                        ],
                        "metadata": {"source": "rating", "score": score}
                    }
                    yield sample
            except Exception:
                continue


def merge_feedback_into_dataset(
    dataset_path: Path,
    output_path: Path,
    include_overrides: bool = True,
    include_ratings: bool = False,
    min_rating: Optional[int] = None,
):
    """Append converted overrides/ratings to an existing dataset (JSONL)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # If dataset_path exists, copy its contents first, otherwise start fresh
    with output_path.open("w", encoding="utf-8") as out:
        if dataset_path and Path(dataset_path).exists():
            with Path(dataset_path).open("r", encoding="utf-8") as orig:
                for line in orig:
                    out.write(line.rstrip("\n") + "\n")

        if include_overrides:
            for s in _iter_overrides():
                out.write(json.dumps(s, ensure_ascii=False) + "\n")

        if include_ratings:
            for s in _iter_ratings(min_rating=min_rating):
                out.write(json.dumps(s, ensure_ascii=False) + "\n")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge dev overrides/ratings into a training dataset")
    parser.add_argument("--dataset", type=Path, help="Existing dataset to append to (optional)", default=None)
    parser.add_argument("--output", type=Path, help="Output dataset path", required=True)
    parser.add_argument("--include-overrides", action="store_true", help="Include /train overrides")
    parser.add_argument("--include-ratings", action="store_true", help="Include /rate ratings")
    parser.add_argument("--min-rating", type=int, help="Only include ratings >= this score")

    args = parser.parse_args()

    ds = args.dataset if args.dataset else None
    merge_feedback_into_dataset(
        ds,
        args.output,
        include_overrides=args.include_overrides,
        include_ratings=args.include_ratings,
        min_rating=args.min_rating,
    )
    print(f"Merged feedback into: {args.output}")
