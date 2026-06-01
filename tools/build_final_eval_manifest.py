#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def resolve_code_relative(value, code_dir):
    path = Path(value)
    if path.is_absolute():
        return path
    return (code_dir / path).resolve()


def sample_key(record):
    return f"{record['category']}::{record['id']}"


def main():
    parser = argparse.ArgumentParser(description="Build a paper-style eval manifest from ASO final_records.")
    parser.add_argument("--code-dir", type=Path, default=Path.cwd())
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=7)
    args = parser.parse_args()

    final_files = sorted(args.save_dir.glob("final_records_rank*.jsonl"))
    if not final_files:
        raise FileNotFoundError(f"No final_records_rank*.jsonl found in {args.save_dir}")

    final_by_key = {}
    duplicate_keys = []
    for path in final_files:
        for record in read_jsonl(path):
            key = sample_key(record)
            if key in final_by_key:
                duplicate_keys.append(key)
                continue

            image_path = resolve_code_relative(record["saved_image"], args.code_dir)
            metadata_path = resolve_code_relative(record["saved_metadata"], args.code_dir)
            if not image_path.exists():
                raise FileNotFoundError(f"Missing final image for {key}: {image_path}")
            if not metadata_path.exists():
                raise FileNotFoundError(f"Missing final metadata for {key}: {metadata_path}")

            final_by_key[key] = {
                "sample_key": key,
                "id": record["id"],
                "category": record["category"],
                "image": str(image_path),
                "text_prompt": record["text_prompt"],
                "query": record["query"],
                "source_image": record.get("source_image") or record.get("image"),
                "saved_metadata": str(metadata_path),
                "training_reward": record.get("reward"),
                "training_success": record.get("success"),
                "final_reason": record.get("final_reason"),
                "attempts": record.get("attempts"),
                "rank": record.get("rank"),
                "global_step": record.get("global_step"),
            }

    baseline_records = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
    ordered = []
    missing = []
    baseline_keys = set()
    for base in baseline_records:
        key = sample_key(base)
        baseline_keys.add(key)
        record = final_by_key.get(key)
        if record is None:
            missing.append(key)
        else:
            ordered.append(record)

    extra = sorted(set(final_by_key) - baseline_keys)
    if missing or extra or duplicate_keys:
        raise RuntimeError(
            "Final records do not align with baseline manifest: "
            f"missing={len(missing)}, extra={len(extra)}, duplicates={len(duplicate_keys)}"
        )
    if len(ordered) != len(baseline_records):
        raise RuntimeError(f"Expected {len(baseline_records)} records, got {len(ordered)}")

    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")

    args.shard_dir.mkdir(parents=True, exist_ok=True)
    shards = [[] for _ in range(args.num_shards)]
    for index, record in enumerate(ordered):
        shards[index % args.num_shards].append(record)
    for index, shard in enumerate(shards):
        shard_path = args.shard_dir / f"shard_{index:02d}.json"
        shard_path.write_text(json.dumps(shard, ensure_ascii=False, indent=2), encoding="utf-8")

    training_success = sum(1 for row in ordered if row.get("training_success") is True)
    print(f"manifest={args.manifest_output}")
    print(f"records={len(ordered)}")
    print(f"training_success={training_success}")
    print(f"training_asr={training_success / len(ordered):.6f}")
    print(f"shards={args.num_shards}")
    for index, shard in enumerate(shards):
        print(f"shard_{index:02d}={len(shard)}")


if __name__ == "__main__":
    main()
