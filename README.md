# Adversarial Style Optimization: Enhancing VLM Jailbreaks by GRPO-based Stylistic Triggers Optimization

<div align="center">

**CVPR 2026 (Oral, Award Candidate)**

[![Paper](https://img.shields.io/badge/Paper-CVPR%202026-blue)](https://openaccess.thecvf.com/content/CVPR2026/html/Luo_Adversarial_Style_Optimization_Enhancing_VLM_Jailbreaks_by_GRPO-based_Stylistic_Triggers_CVPR_2026_paper.html)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

</div>

---

## Overview

![Motivation](assets/intro.png)

Adversarial Style Optimization (ASO) is a plug-and-play enhancement framework for visual jailbreak red-teaming. It targets a stylistic inconsistency in vision-language models: models can often understand the content of an image across visual styles, while their safety alignment can be sensitive to how that content is visually presented.

![Framework](assets/framework.png)

ASO follows a two-stage pipeline:

- **Style Sensitivity Probing**: identify vulnerable style directions from a pool of visual styles.
- **GRPO-based Style Enhancement**: fine-tune an image-editing generator with Dynamic-Batch GRPO and a structurally tiered reward that combines refusal likelihood with HarmBench-based harmfulness feedback.

## Metrics

ASR is the HarmBench binary `Yes` rate. HS is the average HarmBench `success_logprob` (`log P(yes) - log P(no)` in the released scorer).

## Environment Setup

```bash
python3 -m venv ./venvs/train
source ./venvs/train/bin/activate
pip install -U pip
pip install -r requirements/train.txt
pip install -e .
```

For the HarmBench judge service, follow the official [HarmBench](https://github.com/centerforaisafety/HarmBench) setup.

For the victim VLM service, follow the [vLLM OpenAI-compatible server documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).

Copy `.env.example` to `.env`, edit paths and service endpoints, then load it:

```bash
cp .env.example .env
set -a
source .env
set +a
```

## Dataset Preparation

Download the source attack dataset you want to optimize, such as [PKU-Alignment/MM-SafetyBench](https://huggingface.co/datasets/PKU-Alignment/MM-SafetyBench) or [wang021/VLBreakBench](https://huggingface.co/datasets/wang021/VLBreakBench). Set `ASO_DATA_ROOT` to the directory where the datasets are stored.

For the MM-SafetyBench example below, build the ASO manifest from the downloaded image root:

```bash
python tools/build_local_mmsafetybench_sd_typo_manifest.py \
  --root /path/to/MM-SafetyBench-images \
  --output $ASO_DATA_ROOT/MM-SafetyBench/MM-SafetyBench_imgs_clean/mmsafetybench_sd_typo.json
```

## Training

The example training script optimizes ASO on MM-SafetyBench with a vLLM-served victim VLM and a HarmBench judge endpoint.

```bash
source ./venvs/train/bin/activate
export TRAIN_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export ASO_NUM_TRAIN_GPUS=7
export VICTIM_API_BASE_URL=http://<victim-host>:8021/v1
export HARM_BENCH_API_URL=http://<judge-host>:5000
export ASO_TARGET_SAMPLES_PER_EPOCH=48
export WANDB_MODE=offline
bash scripts/run_mmsafetybench_qwen3_train.sh
```

Training outputs are written to `config.save_dir`, which is derived from the selected config and printed when training starts.

## Evaluation

After ASO finishes, evaluate all final images with the victim VLM and HarmBench:

```bash
export VICTIM_ENV_ACTIVATE=/path/to/vllm-env/bin/activate
export VICTIM_MODEL_PATH=/path/to/victim-model
export VICTIM_SERVED_MODEL_NAME=<served-model-name>
export HARM_BENCH_API_URL=http://<judge-host>:5000
export SAVE_DIR=/path/to/aso-training-output
export BASELINE_MANIFEST=$ASO_DATA_ROOT/MM-SafetyBench/MM-SafetyBench_imgs_clean/mmsafetybench_sd_typo.json
bash scripts/run_qr_ours_final_eval_parallel.sh
```

For a single victim-model endpoint:

```bash
export START_EXTRA_QWEN=0
export NUM_SHARDS=1
export QWEN_PORTS=8021
bash scripts/run_qr_ours_final_eval_parallel.sh
```

Results are saved to `outputs/qr_ours_final_eval/<run_id>/summary.json`.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{luo2026adversarial,
  title={Adversarial Style Optimization: Enhancing VLM Jailbreaks by GRPO-based Stylistic Triggers Optimization},
  author={Luo, Bingjun and Guo, Jialin and Yao, Yue and Ding, Xinpeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11--19},
  year={2026}
}
```

## Acknowledgements

This codebase builds on and benefits from several excellent open-source projects, including [yifan123/flow_grpo](https://github.com/yifan123/flow_grpo), [centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench), and the public benchmark resources used by ASO. We thank the authors and maintainers for making their work available to the community.
