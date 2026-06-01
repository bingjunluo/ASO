#!/usr/bin/env python
import argparse
import json
import re
from pathlib import Path


def normalize(text):
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_image_root(root):
    candidates = [
        root / "MM-SafetyBench(imgs)",
        root / "MM-SafetyBench" / "MM-SafetyBench(imgs)",
    ]
    for candidate in candidates:
        nested = candidate / "MM-SafetyBench(imgs)"
        if nested.exists():
            return nested
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find MM-SafetyBench image root under the provided root.")


def build_category_maps(root, image_root):
    source_json_dir = root / "11"
    text_dir = root / "MM-SafetyBench" / "MM-SafetyBench_text"
    if not source_json_dir.exists():
        raise FileNotFoundError(f"Missing source JSON directory: {source_json_dir}")
    if not text_dir.exists():
        raise FileNotFoundError(f"Missing MM-SafetyBench_text directory: {text_dir}")

    source_json = {}
    for path in source_json_dir.glob("*.json"):
        category = re.sub(r"^\d+-", "", path.stem)
        source_json[normalize(category)] = path

    image_dirs = {}
    for path in image_root.iterdir():
        if path.is_dir():
            category = re.sub(r"^\d+-", "", path.name)
            image_dirs[normalize(category)] = path

    text_jsonl = {}
    for path in text_dir.glob("*_SD_TYPO.jsonl"):
        category = path.name.removesuffix("_SD_TYPO.jsonl")
        text_jsonl[normalize(category)] = path

    common = sorted(set(source_json) & set(image_dirs) & set(text_jsonl))
    if not common:
        raise RuntimeError("No categories matched across source JSON, image dirs, and SD_TYPO jsonl files.")
    return common, source_json, image_dirs, text_jsonl


def main():
    parser = argparse.ArgumentParser(description="Build the local MM-SafetyBench SD_TYPO manifest for ASO/baseline eval.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/cvpr_abs"),
        help="Root containing 11/, MM-SafetyBench/, and MM-SafetyBench(imgs)/.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Optional global record limit for smoke tests.")
    parser.add_argument("--relative-to", type=Path, default=None, help="Write image paths relative to this directory.")
    args = parser.parse_args()

    root = args.root
    image_root = find_image_root(root)
    common, source_json, image_dirs, text_jsonl = build_category_maps(root, image_root)

    records = []
    for key in common:
        source_items = json.loads(source_json[key].read_text(encoding="utf-8"))
        text_items = load_jsonl(text_jsonl[key])
        image_dir = image_dirs[key] / "SD_TYPO"
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing SD_TYPO image directory: {image_dir}")

        for item in text_items:
            item_id = str(item["id"])
            meta = source_items[item_id]
            image_path = image_dir / f"{item_id}.jpg"
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image for {text_jsonl[key].name} id={item_id}: {image_path}")
            if args.relative_to:
                image_value = str(image_path.relative_to(args.relative_to))
            else:
                image_value = str(image_path)
            records.append(
                {
                    "id": item_id,
                    "category": re.sub(r"^\d+-", "", image_dirs[key].name),
                    "image": image_value,
                    "text_prompt": item["question"],
                    "query": meta.get("Question") or meta.get("Changed Question"),
                    "changed_question": meta.get("Changed Question"),
                    "key_phrase": meta.get("Key Phrase"),
                    "source": "MM-SafetyBench SD_TYPO",
                }
            )

    if args.limit:
        records = records[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
