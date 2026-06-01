#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize HarmBench-scored JSONL outputs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=1680)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.output_dir.glob("*.jsonl")):
        rows.extend(read_jsonl(path))

    by_key = {}
    duplicates = []
    for row in rows:
        key = row.get("sample_key")
        if key in by_key:
            duplicates.append(key)
            continue
        by_key[key] = row

    unique_rows = list(by_key.values())
    scored = len(unique_rows)
    success = sum(1 for row in unique_rows if bool(row.get("is_harmful")))
    hs_values = [float(row["success_logprob"]) for row in unique_rows if row.get("success_logprob") is not None]
    asr = success / scored if scored else 0.0
    avg_hs = sum(hs_values) / scored if scored else 0.0

    summary = {
        "raw_rows": len(rows),
        "scored": scored,
        "expected": args.expected,
        "missing": max(0, args.expected - scored),
        "duplicates": len(duplicates),
        "success": success,
        "asr": asr,
        "asr_percent": asr * 100.0,
        "avg_hs": avg_hs,
        "hs_count": len(hs_values),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"scored={scored}/{args.expected}")
    print(f"success={success}")
    print(f"ASR={asr * 100.0:.4f}%")
    print(f"avg_HS={avg_hs:.6f}")
    print(f"raw_rows={len(rows)} duplicates={len(duplicates)} missing={summary['missing']}")
    print(f"summary={args.summary}")


if __name__ == "__main__":
    main()
