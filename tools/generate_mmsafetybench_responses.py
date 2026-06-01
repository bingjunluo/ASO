#!/usr/bin/env python
import argparse
import base64
import json
import mimetypes
from pathlib import Path

import requests


def sample_key(record, index):
    if record.get("sample_key"):
        return str(record["sample_key"])
    category = record.get("category", "unknown")
    item_id = record.get("id", index)
    return f"{category}::{item_id}"


def resolve_image_path(image_value, manifest_path):
    image_path = Path(image_value)
    if image_path.is_absolute():
        return image_path
    return (manifest_path.parent / image_path).resolve()


def encode_image(path):
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def image_data_url(path):
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime_type};base64,{encode_image(path)}"


def call_victim(base_url, api_key, model, image_path, prompt, timeout, max_tokens, temperature):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def load_done(path):
    done = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["sample_key"])
    return done


def main():
    parser = argparse.ArgumentParser(description="Generate Qwen3-VL responses for MM-SafetyBench QR/SD_TYPO.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--victim-base-url", default="http://localhost:8021/v1")
    parser.add_argument("--victim-api-key", default="EMPTY")
    parser.add_argument("--victim-model", default="qwen3-vl-8b")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.limit:
        records = records[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.output) if args.resume else set()
    mode = "a" if args.resume else "w"

    generated = 0
    with args.output.open(mode, encoding="utf-8") as handle:
        for index, record in enumerate(records):
            key = sample_key(record, index)
            if key in done:
                continue

            image_path = resolve_image_path(record["image"], args.manifest)
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image for {key}: {image_path}")

            response_text = call_victim(
                args.victim_base_url,
                args.victim_api_key,
                args.victim_model,
                image_path,
                record["text_prompt"],
                args.timeout,
                args.max_tokens,
                args.temperature,
            )
            out = {
                "sample_key": key,
                "index": index,
                "id": record.get("id"),
                "category": record.get("category"),
                "image": str(image_path),
                "text_prompt": record["text_prompt"],
                "query": record["query"],
                "response": response_text,
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            handle.flush()
            generated += 1
            print(f"[{index + 1}/{len(records)}] generated={generated} sample_key={key}")

    print(f"Finished. New generated={generated}, output={args.output}")


if __name__ == "__main__":
    main()
