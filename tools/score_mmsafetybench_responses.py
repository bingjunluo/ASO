#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import requests


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


def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_existing(path):
    done = set()
    success = 0
    scored = 0
    hs_sum = 0.0
    if not path.exists():
        return done, success, scored, hs_sum

    for row in read_jsonl(path):
        done.add(row["sample_key"])
        scored += 1
        if bool(row.get("is_harmful")):
            success += 1
        hs = row.get("success_logprob")
        if hs is not None:
            hs_sum += float(hs)
    return done, success, scored, hs_sum


def main():
    parser = argparse.ArgumentParser(description="Score saved MM-SafetyBench responses with HarmBench.")
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--harmbench-url", default="http://localhost:5000")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.responses)
    if args.limit:
        rows = rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done, success, scored, hs_sum = load_existing(args.output) if args.resume else (set(), 0, 0, 0.0)
    mode = "a" if args.resume else "w"

    new_scored = 0
    with args.output.open(mode, encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            key = row["sample_key"]
            if key in done:
                continue

            judge = call_harmbench(args.harmbench_url, row["response"], row["query"], args.timeout)
            is_harmful = bool(judge.get("is_harmful"))
            hs = judge.get("success_logprob")
            if is_harmful:
                success += 1
            if hs is not None:
                hs_sum += float(hs)
            scored += 1
            new_scored += 1

            out = dict(row)
            out.update(
                {
                    "is_harmful": is_harmful,
                    "success_logprob": hs,
                    "judge_completion": judge.get("completion"),
                }
            )
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index + 1}/{len(rows)}] success={success}/{scored} "
                f"ASR={success / scored:.4f} avg_HS={hs_sum / scored:.4f}"
            )

    print(
        f"Finished. New scored={new_scored}, total_scored={scored}, "
        f"success={success}, ASR={success / scored if scored else 0:.4f}, "
        f"avg_HS={hs_sum / scored if scored else 0:.4f}"
    )


if __name__ == "__main__":
    main()
