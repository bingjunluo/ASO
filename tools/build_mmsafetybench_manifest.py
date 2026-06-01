#!/usr/bin/env python
import argparse
import json
import random
from pathlib import Path


IMAGE_KEYS = ("image", "image_path", "img", "img_path", "file_name", "filename", "path")
QUERY_KEYS = ("query", "behavior", "harmful_behavior", "goal", "target", "instruction")
PROMPT_KEYS = ("text_prompt", "prompt", "question", "query", "instruction")


def load_records(path):
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(flatten(data))


def flatten(value):
    if isinstance(value, list):
        for item in value:
            yield from flatten(item)
    elif isinstance(value, dict):
        if any(key in value for key in IMAGE_KEYS):
            yield value
        else:
            for item in value.values():
                yield from flatten(item)


def first_present(record, keys):
    for key in keys:
        value = record.get(key)
        if value:
            return value
    return None


def convert_record(record, image_prefix):
    image = first_present(record, IMAGE_KEYS)
    query = first_present(record, QUERY_KEYS)
    text_prompt = first_present(record, PROMPT_KEYS) or query
    if not image or not query or not text_prompt:
        return None
    image = str(Path(image_prefix) / image) if image_prefix else str(image)
    converted = dict(record)
    converted.update(
        {
            "image": image,
            "query": str(query),
            "text_prompt": str(text_prompt),
        }
    )
    return converted


def main():
    parser = argparse.ArgumentParser(description="Build an ASO manifest from MM-SafetyBench-style JSON/JSONL.")
    parser.add_argument("--input", required=True, type=Path, help="Raw benchmark JSON or JSONL.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON manifest consumed by ASO.")
    parser.add_argument("--image-prefix", default="", help="Optional prefix prepended to relative image paths.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of records for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = [convert_record(record, args.image_prefix) for record in load_records(args.input)]
    records = [record for record in records if record is not None]
    if args.limit:
        random.Random(args.seed).shuffle(records)
        records = records[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
