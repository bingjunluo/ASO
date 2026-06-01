#!/usr/bin/env python
import argparse
import base64
import json
from pathlib import Path

import requests


def resolve_image_path(image_value, manifest_path):
    image_path = Path(image_value)
    if image_path.is_absolute():
        return image_path
    return (manifest_path.parent / image_path).resolve()


def encode_image(path):
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def call_victim(base_url, api_key, model, image_path, prompt, timeout):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"},
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 512,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def call_harmbench(base_url, response_text, behavior, timeout):
    payload = {
        "generations": response_text,
        "behavior": behavior,
        "device": "cuda",
        "return_logprob": True,
    }
    response = requests.post(f"{base_url.rstrip('/')}/predict", json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    message = str(data.get("message", ""))
    if data.get("is_harmful") is None and message.startswith("prediction failed"):
        raise RuntimeError(message)
    return data


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL baseline on a prepared MM-SafetyBench manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--victim-base-url", default="http://localhost:8021/v1")
    parser.add_argument("--victim-api-key", default="EMPTY")
    parser.add_argument("--victim-model", default="qwen3-vl-8b")
    parser.add_argument("--harmbench-url", default="http://localhost:5000")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.limit:
        records = records[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    success = 0
    scored = 0
    hs_sum = 0.0
    if args.resume and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add(row["sample_key"])
                scored += 1
                if bool(row.get("is_harmful")):
                    success += 1
                hs = row.get("success_logprob")
                if hs is not None:
                    hs_sum += float(hs)

    new_scored = 0
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for idx, record in enumerate(records):
            sample_key = f"{record['category']}::{record['id']}"
            if sample_key in done:
                continue
            image_path = resolve_image_path(record["image"], args.manifest)
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image for {sample_key}: {image_path}")
            response_text = call_victim(
                args.victim_base_url,
                args.victim_api_key,
                args.victim_model,
                image_path,
                record["text_prompt"],
                args.timeout,
            )
            judge = call_harmbench(args.harmbench_url, response_text, record["query"], args.timeout)
            is_harmful = bool(judge.get("is_harmful"))
            hs = judge.get("success_logprob")
            if is_harmful:
                success += 1
            if hs is not None:
                hs_sum += float(hs)
            scored += 1
            new_scored += 1
            out = {
                "sample_key": sample_key,
                "index": idx,
                "id": record["id"],
                "category": record["category"],
                "image": record["image"],
                "text_prompt": record["text_prompt"],
                "query": record["query"],
                "response": response_text,
                "is_harmful": is_harmful,
                "success_logprob": hs,
                "judge_completion": judge.get("completion"),
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{idx + 1}/{len(records)}] success={success}/{scored} "
                f"ASR={success / scored:.4f} avg_HS={hs_sum / scored:.4f}"
            )

    print(
        f"Finished. New scored={new_scored}, total_scored={scored}, "
        f"success={success}, ASR={success / scored if scored else 0:.4f}, "
        f"avg_HS={hs_sum / scored if scored else 0:.4f}"
    )


if __name__ == "__main__":
    main()
