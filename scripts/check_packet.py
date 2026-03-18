#!/usr/bin/env python3
"""Validate generated ChemRxiv packet JSONL schema and counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FIELDS = {
    "query_id",
    "query_text",
    "ground_truth_doc_id",
    "ground_truth_score",
    "ground_truth_text",
    "top_k",
    "retrieved",
    "success",
}

# Enforce the exact structure requested (no titles anywhere)
FORBIDDEN_TOP_LEVEL_FIELDS = {"ground_truth_title"}
FORBIDDEN_RETRIEVED_FIELDS = {"title"}
REQUIRED_RETRIEVED_FIELDS = {"rank", "score", "doc_id", "text"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to JSONL packet")
    ap.add_argument("--expected-success", type=int, default=25)
    ap.add_argument("--expected-failure", type=int, default=25)
    ap.add_argument("--expected-top-k", type=int, default=10)
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        raise SystemExit(f"ERROR: file not found: {p}")

    success = 0
    failure = 0
    n = 0
    meta = None

    with p.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            if "__meta__" in obj:
                if meta is not None:
                    raise SystemExit(f"ERROR: multiple meta lines found (line {ln})")
                meta = obj["__meta__"]
                continue

            missing = REQUIRED_FIELDS - set(obj.keys())
            if missing:
                raise SystemExit(f"ERROR line {ln}: missing fields {sorted(missing)}")

            forbidden = FORBIDDEN_TOP_LEVEL_FIELDS & set(obj.keys())
            if forbidden:
                raise SystemExit(f"ERROR line {ln}: forbidden top-level fields present {sorted(forbidden)}")

            if not isinstance(obj["retrieved"], list):
                raise SystemExit(f"ERROR line {ln}: 'retrieved' must be a list")

            for j, r in enumerate(obj["retrieved"], start=1):
                if not isinstance(r, dict):
                    raise SystemExit(f"ERROR line {ln}: retrieved[{j}] must be an object")
                missing_r = REQUIRED_RETRIEVED_FIELDS - set(r.keys())
                if missing_r:
                    raise SystemExit(f"ERROR line {ln}: retrieved[{j}] missing fields {sorted(missing_r)}")
                forbidden_r = FORBIDDEN_RETRIEVED_FIELDS & set(r.keys())
                if forbidden_r:
                    raise SystemExit(f"ERROR line {ln}: retrieved[{j}] forbidden fields present {sorted(forbidden_r)}")

            if len(obj["retrieved"]) != args.expected_top_k:
                raise SystemExit(
                    f"ERROR line {ln}: retrieved length {len(obj['retrieved'])} != expected_top_k {args.expected_top_k}"
                )

            top_k = int(obj["top_k"])
            if top_k != args.expected_top_k:
                raise SystemExit(f"ERROR line {ln}: top_k={top_k} != expected_top_k={args.expected_top_k}")

            if bool(obj["success"]):
                success += 1
            else:
                failure += 1
            n += 1

    if meta is None:
        raise SystemExit("ERROR: missing __meta__ line")

    if success != args.expected_success or failure != args.expected_failure:
        raise SystemExit(
            "ERROR: count mismatch "
            f"success={success} (expected {args.expected_success}), "
            f"failure={failure} (expected {args.expected_failure})"
        )

    print("Packet validation OK")
    print(f"Examples: {n}")
    print(f"Success: {success}")
    print(f"Failure: {failure}")


if __name__ == "__main__":
    main()
