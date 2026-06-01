#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoProcessor


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


def load_done(path):
    done = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["sample_key"])
    return done


def load_qwen3_vl(model_name, dtype, device_map, attn_implementation):
    try:
        from transformers import Qwen3VLForConditionalGeneration

        model_cls = Qwen3VLForConditionalGeneration
    except ImportError:
        from transformers import AutoModelForImageTextToText

        model_cls = AutoModelForImageTextToText

    dtype_value = "auto" if dtype == "auto" else getattr(torch, dtype)
    kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    try:
        return model_cls.from_pretrained(model_name, dtype=dtype_value, **kwargs)
    except TypeError:
        return model_cls.from_pretrained(model_name, torch_dtype=dtype_value, **kwargs)


def generate_one(model, processor, image_path, prompt, max_new_tokens):
    # Keep content order aligned with the OpenAI/vLLM baseline route.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": str(image_path)},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


def main():
    parser = argparse.ArgumentParser(description="Generate MM-SafetyBench responses with local Transformers Qwen3-VL.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("VICTIM_MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.limit:
        records = records[: args.limit]

    print(f"Loading processor: {args.model}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    print(f"Loading model: {args.model}", flush=True)
    model = load_qwen3_vl(args.model, args.dtype, args.device_map, args.attn_implementation or None)
    model.eval()

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

            response_text = generate_one(model, processor, image_path, record["text_prompt"], args.max_new_tokens)
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
            print(f"[{index + 1}/{len(records)}] generated={generated} sample_key={key}", flush=True)

    print(f"Finished. New generated={generated}, output={args.output}")


if __name__ == "__main__":
    main()
