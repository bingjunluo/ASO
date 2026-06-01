# 7 GPU
## vlbreakbench_ideator
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=7 --main_process_port 29501 \
    scripts/train_flux_kontext_search_dynamic.py \
    --config config/grpo_logprob.py:complex_logprob_flux_kontext_vlbreakbench_ideator_qwen3_dynamic

accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=7 --main_process_port 29501 \
    scripts/train_flux_kontext_search_dynamic.py \
    --config config/grpo_logprob.py:complex_logprob_flux_kontext_vlbreakbench_ideator_llava_ov_dynamic

## mmsafetybench_qr_attack
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=7 --main_process_port 29501 \
    scripts/train_flux_kontext_search_dynamic.py \
    --config config/grpo_logprob.py:complex_logprob_flux_kontext_mmsafetybench_qr_qwen3_vl_dynamic

accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=7 --main_process_port 29501 \
    scripts/train_flux_kontext_search_dynamic.py \
    --config config/grpo_logprob.py:complex_logprob_flux_kontext_mmsafetybench_qr_llava_ov_dynamic
