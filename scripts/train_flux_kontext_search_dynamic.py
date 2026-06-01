from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import FluxKontextPipeline
from diffusers.utils.torch_utils import is_compiled_module
from transformers.integrations.deepspeed import (
    is_deepspeed_zero3_enabled,
    set_hf_deepspeed_config,
    unset_hf_deepspeed_config,
)
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
# 【修改】移除了 PerPromptStatTracker，因为新流程不再需要它
# from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.flux_kontext_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_flux import encode_prompt
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from pathlib import Path
from torch.utils.data import Dataset # Sampler, DataLoader 已不再需要
from flow_grpo.ema import EMAModuleWrapper

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


def _prompt_key(item):
    if "prompt_with_image_path" in item:
        return item["prompt_with_image_path"]
    return f"{item['prompt']}_{item['metadata']['image']}"


def _safe_filename_part(value, default="item", max_len=80):
    text = str(value or default)
    text = text.replace("\\", "_").replace("/", "_").replace(":", "_")
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)
    safe = safe.strip("._") or default
    return safe[:max_len]


def _metadata_identity(metadata):
    category = _safe_filename_part(metadata.get("category") or metadata.get("source") or "sample")
    source = _safe_filename_part(metadata.get("source") or "unknown")
    item_id = _safe_filename_part(metadata.get("id") or Path(str(metadata.get("image", "image"))).stem)
    digest_source = "|".join(
        str(metadata.get(key, "")) for key in ("image", "query", "text_prompt", "changed_question")
    )
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:12]
    return category, source, item_id, digest


def _success_output_stem(metadata, rank, global_step, local_index, prefix="rank"):
    category, source, item_id, digest = _metadata_identity(metadata)
    return (
        f"{prefix}{rank:02d}__step{int(global_step):06d}__idx{int(local_index):02d}"
        f"__{category}__{source}__{item_id}__{digest}"
    )


def _append_jsonl(save_dir, filename, record):
    os.makedirs(save_dir, exist_ok=True)
    output = os.path.join(save_dir, filename)
    with open(output, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_success_record(save_dir, rank, record):
    _append_jsonl(save_dir, f"success_records_rank{rank:02d}.jsonl", record)


def _append_final_record(save_dir, rank, record):
    _append_jsonl(save_dir, f"final_records_rank{rank:02d}.jsonl", record)


def _write_final_record(
    save_dir,
    rank,
    global_step,
    local_index,
    prompt_info,
    image_array,
    reward,
    attempts,
    success,
    final_reason,
    success_threshold,
):
    pil_image = Image.fromarray(image_array)
    metadata = prompt_info["metadata"]
    base_filename = _success_output_stem(metadata, rank, global_step, local_index, prefix="final__rank")
    image_output = os.path.join(save_dir, f"{base_filename}.jpg")
    metadata_output = os.path.join(save_dir, f"{base_filename}.json")
    pil_image.save(image_output)
    record = dict(metadata)
    record.update(
        {
            "saved_image": image_output,
            "saved_metadata": metadata_output,
            "source_image": metadata.get("image"),
            "rank": rank,
            "global_step": int(global_step),
            "local_index": int(local_index),
            "attempts": int(attempts),
            "reward": float(reward),
            "success": bool(success),
            "final_reason": final_reason,
            "success_threshold": float(success_threshold),
        }
    )
    with open(metadata_output, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    _append_final_record(save_dir, rank, record)
    return record


class GenevalPromptImageDataset(Dataset):
    def __init__(self, dataset, prompt, split='train'):
        # dataset = '../dataset/VLBreakBench/vlbreakbench_challenge_fully_sampled.json'
        # self.dataset取VLBreakBench
        self.prompt = prompt
        self.dataset = os.path.dirname(dataset)
        self.file_path = dataset
        # self.dataset = dataset.split
        # self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = json.load(f)
        
        # self.metadatas = random.choices(self.metadatas, k=30)
        
        # if split == 'train':
        #     self.metadatas = random.choices(self.metadatas, k=128)
        # if split == 'test':
        #     self.metadatas = random.choices(self.metadatas, k=128)
            # self.metadatas = [json.loads(line) for line in f]
            # self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.metadatas)
    
    def __getitem__(self, idx):
        item = {
            "prompt": self.prompt,
            "metadata": self.metadatas[idx]
        }
        # Assuming 'image' in metadata contains a path to the image file
        image_path = self.metadatas[idx]['image']
        item["prompt_with_image_path"] = f"{self.prompt}_{image_path}"
        image_path = image_path if os.path.isabs(image_path) else os.path.join(self.dataset, image_path)
        image = Image.open(image_path).convert('RGB')
        item["image"] = image
        return item

# 【删除】DistributedKRepeatSampler 类不再需要，已被删除

# 【保留】辅助函数

def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        text_ids = text_ids.to(device)
    return prompt_embeds, pooled_prompt_embeds

# 【删除】calculate_zero_std_ratio 函数不再需要，已被删除
# 【删除】create_generator 函数不再需要，已被删除

# 【保留】核心计算函数
        
def compute_log_prob(transformer, pipeline, sample, j, config):
    latents = sample["latents"][:, j]
    device = latents.device
    dtype = latents.dtype
    if transformer.module.config.guidance_embeds:
        guidance = torch.tensor([config.sample.guidance_scale], device=device)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    # Predict the noise residual
    latent_model_input = sample["latents"][:, j]
    if sample["image_latents"] is not None:
        latent_model_input = torch.cat([latent_model_input, sample["image_latents"]],dim=1)
    model_pred = transformer(
        hidden_states=latent_model_input,
        timestep=sample["timesteps"][:, j] / 1000,
        guidance=guidance,
        pooled_projections=sample["pooled_prompt_embeds"],
        encoder_hidden_states=sample["prompt_embeds"],
        txt_ids= sample["text_ids"][0],
        img_ids=sample["latent_ids"][0],
        return_dict=False,
    )[0]
    model_pred = model_pred[:, : sample["latents"][:, j].size(1)]
    # compute the log prob of next_latents given latents under the current model
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        model_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t

# 【删除】eval 函数不再需要，已被删除

# 【保留】工具函数

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

# 【删除】save_ckpt 函数不再需要，已被删除

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        # log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:
        wandb.init(
            project="flow_grpo",
            config=config.to_dict(),
            # mode="disabled"
        )
        # accelerator.init_trackers(
        #     project_name="flow-grpo",
        #     config=config.to_dict(),
        #     init_kwargs={"wandb": {"name": config.run_name}},
        # )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = FluxKontextPipeline.from_pretrained(
        config.pretrained.model,
        low_cpu_mem_usage=False
    )
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    
    pipeline.transformer.to(accelerator.device)

    if config.use_lora:
        # Set correct lora layers
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
            "proj_mlp",
        ]
        transformer_lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
    
    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    accelerator.print("正在加载完整数据集到内存...")
    full_dataset = GenevalPromptImageDataset(config.dataset, config.prompt, 'train')

    D_pending = [{"prompt": item["prompt"], "image": item["image"], "metadata": item["metadata"]} for item in full_dataset]
    logger.info(f"成功加载 {len(D_pending)} 个 prompts 作为总待办清单。")

    # 【✅ 新代码 - 采用更公平、更简洁的 Pythonic 写法】
    # a. 每个进程获取自己的数据分片 (公平版本)
    # 这行代码会自动地、交错地将数据分配给所有进程
    D_pending_local = D_pending[accelerator.process_index::accelerator.num_processes]
    num_local_prompts = len(D_pending_local)

    # b. 全局通信，找到最长的数据列表有多长
    local_prompts_tensor = torch.tensor([num_local_prompts], device=accelerator.device)
    all_prompts_tensor = accelerator.gather(local_prompts_tensor)
    max_prompts = all_prompts_tensor.max().item()

    # c. 对短的列表进行填充
    padding_needed = max_prompts - num_local_prompts
    if padding_needed > 0 and num_local_prompts > 0:
        # 从已有的真实数据中随机采样来作为填充
        padding_data = random.choices(D_pending_local, k=padding_needed)
        D_pending_local.extend(padding_data)
    
    # d. 如果一个进程没有任何数据，则用其他进程的数据填充（非常罕见的边缘情况）
    # (此部分逻辑可以根据需要添加，但通常数据分配不会如此极端)

    logger.info(
        f"进程 {accelerator.process_index}: "
        f"原始 prompts={num_local_prompts}, "
        f"填充后 prompts={len(D_pending_local)}. "
        f"全局最大 prompts 数为 {max_prompts}."
    )
    accelerator.wait_for_everyone() # 确保所有进程都完成了填充

    # 3. 基于填充后的数据创建批次
    batch_size = config.sample.train_batch_size
    local_batches = [
        D_pending_local[i:i + batch_size] 
        for i in range(0, len(D_pending_local), batch_size)
    ]
    num_total_batches = len(local_batches) # 现在所有进程的 num_total_batches 都一样了

    if accelerator.is_main_process:
        if os.path.exists(config.save_dir):
            import shutil
            shutil.rmtree(config.save_dir)
        os.makedirs(config.save_dir, exist_ok=True)
        logger.info(f"成功生成的图片将保存到: {config.save_dir}")

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # for deepspeed zero
    if accelerator.state.deepspeed_plugin:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.sample.train_batch_size
    # prepare prompt and reward fn
    if is_deepspeed_zero3_enabled():
        # Using deepspeed zero3 will cause the model parameter `weight.shape` to be empty.
        unset_hf_deepspeed_config()
        reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
        eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
        set_hf_deepspeed_config(accelerator.state.deepspeed_plugin.dschf)
    else:
        reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
        eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    
    # Prepare everything with our `accelerator`.
    transformer, optimizer = accelerator.prepare(transformer, optimizer)
    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)


    global_step = 0
    pending_iterator = iter(D_pending_local)
    
    total_prompts = len(D_pending_local)
    global_total_prompts = len(D_pending)
    global_succeeded_prompts = 0
    global_processed_prompts = 0

    
    # 动态B_target相关变量
    max_attempts_per_sample = config.train.max_attempts_per_sample
    dynamic_batch_size = config.sample.train_batch_size  # 动态批次大小
    
    # 初始化B_target为第一批数据
    B_target = []
    try:
        for _ in range(dynamic_batch_size):
            B_target.append(next(pending_iterator))
    except StopIteration:
        # 如果数据不足，用已有数据填充
        while len(B_target) < dynamic_batch_size:
            B_target.append(random.choice(D_pending_local) if D_pending_local else None)
        B_target = [item for item in B_target if item is not None]
    
    # 为每个样本跟踪尝试次数和连续成功次数
    sample_attempts = {_prompt_key(item): 0 for item in B_target}
    sample_consecutive_successes = {_prompt_key(item): 0 for item in B_target}
    finalized_prompt_keys = set()
    
    # 计算总批次数（基于总数据量）
    num_total_batches = (len(D_pending_local) + dynamic_batch_size - 1) // dynamic_batch_size
    
    # 2. 使用固定轮数的 for 循环
    with tqdm(total=len(D_pending_local), desc=f"全局搜索进度 (进程 {accelerator.process_index})", disable=not accelerator.is_local_main_process) as pbar:
        while B_target:  # 当B_target不为空时继续循环
            # 检查当前进程是否还有任务
            # if len(B_target) < config.sample.train_batch_size:
            #     # 这个进程的任务已经完成了，进入“空转”模式
            #     # 它必须在这里等待，直到其他进程也完成这一轮
            #     accelerator.wait_for_everyone()
            #     pbar.update(1)
            #     continue # 直接进入下一轮 for 循环

            # # 如果还有任务，正常组建批次
            # B_target = []
            # try:
            #     for _ in range(config.sample.train_batch_size):
            #         B_target.append(next(pending_iterator))
            # except StopIteration:
            #     # 这通常不应该发生，因为我们已经通过 batch_idx 控制了循环
            #     # 但作为保险措施
            #     pass

            # if not B_target:
            #     accelerator.wait_for_everyone()
            #     pbar.update(1)
            #     continue

            early_stopping_counter = 0

            initial_batch_size = len(B_target)
            logger.info(f"进程 {accelerator.process_index} 开始处理批次，初始大小: {initial_batch_size}")

            # 🚀 1. 初始化一个与批次大小相同的“成功掩码”
            succeeded_mask = torch.zeros(len(B_target), dtype=torch.bool, device=accelerator.device)

            # 2. 内部优化循环
            # for inner_step in tqdm(range(config.train.max_inner_steps), desc=f"优化批次 (剩余 {len(B_target)})", leave=False, disable=not accelerator.is_local_main_process):
                # if not B_target:
                #     accelerator.print(f"批次内所有 prompts 已攻克，提前进入下一批次。")
                #     break

            current_prompts = [item["prompt"] for item in B_target]
            current_metadata = [item["metadata"] for item in B_target]
            ref_images = [item["image"] for item in B_target]

            logger.info(f'进程 {accelerator.process_index} A. 生成与评估')
            pipeline.transformer.eval()

            # 处理图片分割逻辑
            processed_ref_images = []
            image_parts = []  # 存储每个图片的分割信息
            
            for ref_image in ref_images:
                height, width = ref_image.height, ref_image.width
                new_width = config.resolution
                new_height = int(new_width * (height / width))
                ref_image = ref_image.resize((new_width, new_height))
                
                if height > new_width:
                    # 分割图片为两部分：<=512的部分和>512的部分
                    
                    if 'HMRD' in full_dataset.file_path:
                        part1 = ref_image.crop((0, new_height - new_width, new_width, new_height))
                        part2 = ref_image.crop((0, 0, new_width, new_height - new_width))
                    else:
                        part1 = ref_image.crop((0, 0, new_width, new_width))
                        part2 = ref_image.crop((0, new_width, new_width, new_height))
                    
                    processed_ref_images.append(part1)
                    image_parts.append({'has_part2': True, 'part1': part1, 'part2': part2})
                else:
                    # 高度<=512，直接resize整个图片
                    processed_ref_images.append(ref_image)
                    image_parts.append({'has_part2': False, 'part1': ref_image, 'part2': None})
            
            ref_images = processed_ref_images

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                current_prompts, 
                text_encoders, 
                tokenizers, 
                max_sequence_length=128, 
                device=accelerator.device
            )

            with torch.no_grad():
                images, latents, latent_ids, text_ids, log_probs, image_latents = pipeline_with_logprob(
                    pipeline,
                    image=ref_images,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution, 
                    max_area=config.resolution*config.resolution,
                    noise_level=config.sample.noise_level,
                    # generator=generator
                )

            # 合并图片：将生成的结果与第二部分图片合并
            merged_images = []
            for i, (img, parts_info) in enumerate(zip(images, image_parts)):
                if parts_info['has_part2']:
                    # 将生成的图片转换为PIL格式
                    if isinstance(img, torch.Tensor):
                        img_np = (img * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                        img_np = img_np.transpose(1, 2, 0)
                        generated_part = Image.fromarray(img_np)
                    else:
                        generated_part = img

                    
                    # 获取第二部分图片并resize到与第一部分相同的宽度
                    part2 = parts_info['part2']
                    
                    # 合并两部分图片
                    merged_width = generated_part.width
                    merged_height = generated_part.height + part2.height
                    
                    merged_img = Image.new('RGB', (merged_width, merged_height))

                    if 'HMRD' in full_dataset.file_path:
                        merged_img.paste(part2, (0, 0))
                        merged_img.paste(generated_part, (0, part2.height))
                    else:
                        merged_img.paste(generated_part, (0, 0))
                        merged_img.paste(part2, (0, generated_part.height))
                    
                    # 将合并后的图片转换回tensor格式
                    # merged_img_np = np.array(merged_img).astype(np.float32) / 255.0
                    # merged_img_np = merged_img_np.transpose(2, 0, 1)  # 转为CHW
                    # merged_img_tensor = torch.from_numpy(merged_img_np)
                    merged_images.append(np.array(merged_img))
                else:
                    if isinstance(img, torch.Tensor):
                        img_np = (img * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                        img_np = img_np.transpose(1, 2, 0)
                        generated_part = Image.fromarray(img_np)
                    else:
                        generated_part = img
                    merged_images.append(np.array(generated_part))
        
            # compute rewards asynchronously
            rewards_future = executor.submit(reward_fn, merged_images, current_prompts, current_metadata, config.victim, ref_images, only_strict=True)
        
            logger.info(f'进程 {accelerator.process_index} --- B. 更新成功掩码并保存图片 ---')
            rewards, _ = rewards_future.result()
            rewards_tensor = torch.tensor(rewards['avg'], device=accelerator.device)
            success_threshold = config.success_threshold
            
            # 找出本次成功的样本
            current_success_mask = (rewards_tensor > success_threshold)
            
            # 更新连续成功次数
            for i, item in enumerate(B_target):
                prompt_key = _prompt_key(item)
                if current_success_mask[i]:
                    sample_consecutive_successes[prompt_key] += 1
                else:
                    sample_consecutive_successes[prompt_key] = 0
            
            # 找出连续K次成功的样本（K=3）
            consecutive_k_success_mask = torch.zeros(len(B_target), dtype=torch.bool, device=accelerator.device)
            for i, item in enumerate(B_target):
                prompt_key = _prompt_key(item)
                if sample_consecutive_successes[prompt_key] >= config.train.consecutive_success:  # 连续3次成功
                    consecutive_k_success_mask[i] = True
            
            # 找出本次新达到连续K次成功的样本
            newly_consecutive_success_mask = consecutive_k_success_mask #& (~succeeded_mask)
            
            if newly_consecutive_success_mask.any():
                # 更新总的成功掩码
                succeeded_mask |= newly_consecutive_success_mask
                
                # 保存新达到连续K次成功的图片
                succeeded_indices = torch.where(newly_consecutive_success_mask)[0]
                logger.info(f"内部步骤: 新达到连续3次成功的样本 {len(succeeded_indices)} 个！")
                for i in succeeded_indices.tolist():
                    prompt_info = B_target[i]
                    image_array = merged_images[i]
                    pil_image = Image.fromarray(image_array)
                    metadata = prompt_info['metadata']
                    base_filename = _success_output_stem(metadata, accelerator.process_index, global_step, i)
                    image_output = os.path.join(config.save_dir, f'{base_filename}.jpg')
                    metadata_output = os.path.join(config.save_dir, f'{base_filename}.json')
                    pil_image.save(image_output)
                    record = dict(metadata)
                    record.update({
                        "saved_image": image_output,
                        "saved_metadata": metadata_output,
                        "source_image": metadata.get("image"),
                        "rank": accelerator.process_index,
                        "global_step": global_step,
                        "local_index": i,
                        "reward": float(rewards_tensor[i].detach().cpu().item()),
                    })
                    with open(metadata_output, 'w', encoding='utf-8') as f:
                        json.dump(record, f, ensure_ascii=False, indent=2)
                    _append_success_record(config.save_dir, accelerator.process_index, record)

            logger.info(f'进程 {accelerator.process_index} --- C. 模型训练 (使用掩码) ---')
            pipeline.transformer.train()
            
            stacked_latents = torch.stack(latents, dim=1)
            stacked_log_probs = torch.stack(log_probs, dim=1)
            
            advantages_tensor = rewards_tensor.clone() # 使用当前步骤的 reward/advantage
            
            # 🚀 关键步骤：应用损失屏蔽！
            # 将已成功样本的 advantage 设为 0，这样它们的梯度贡献也为 0
            # advantages_tensor[succeeded_mask] = 0.0 
            
            advantages_tensor = advantages_tensor.unsqueeze(1).repeat(1, num_train_timesteps)

            sample = {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "latent_ids": latent_ids.unsqueeze(0).repeat(len(current_prompts),1,1),
                "image_latents": image_latents,
                "text_ids": text_ids.unsqueeze(0).repeat(len(current_prompts),1,1),
                "timesteps": pipeline.scheduler.timesteps.repeat(len(current_prompts), 1),
                "latents": stacked_latents[:, :-1],
                "next_latents": stacked_latents[:, 1:],
                "log_probs": stacked_log_probs[:],
                "advantages": advantages_tensor,
            }
            info = defaultdict(list)
            for j in range(num_train_timesteps):
                # DeepSpeed ZeRO-2 cannot enter Accelerator's model-specific no_sync path.
                with accelerator.accumulate():
                    with autocast():
                        prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(transformer, pipeline, sample, j, config)
                        if config.train.beta > 0:
                            with torch.no_grad():
                                with unwrap_model(transformer, accelerator).disable_adapter():
                                    _, _, prev_sample_mean_ref, _ = compute_log_prob(transformer, pipeline, sample, j, config)

                    # grpo logic
                    advantages = torch.clamp(
                        sample["advantages"][:, j],
                        -config.train.adv_clip_max,
                        config.train.adv_clip_max,
                    )
                    ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                    unclipped_loss = -advantages * ratio
                    clipped_loss = -advantages * torch.clamp(
                        ratio,
                        1.0 - config.train.clip_range,
                        1.0 + config.train.clip_range,
                    )
                    policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                    if config.train.beta > 0:
                        kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2), keepdim=True) / (2 * std_dev_t ** 2)
                        kl_loss = torch.mean(kl_loss)
                        loss = policy_loss + config.train.beta * kl_loss
                        info["kl_loss"].append(kl_loss)
                    else:
                        loss = policy_loss

                    info["policy_loss"].append(policy_loss)

                    info["loss"].append(loss)

                    # backward pass
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            transformer.parameters(), config.train.max_grad_norm
                        )
                    optimizer.step()
                    optimizer.zero_grad()
            
            if config.train.ema and accelerator.sync_gradients:
                ema.step(transformer_trainable_parameters, global_step)
            
            global_newly_succeeded_sum = accelerator.reduce(newly_consecutive_success_mask.sum(), reduction="sum")
            global_succeeded_sum = accelerator.reduce(succeeded_mask.sum(), reduction="sum")
            # 同步所有进程的global_processed_prompts
            local_processed_tensor = torch.tensor([global_processed_prompts], device=accelerator.device)
            global_processed_tensor = accelerator.reduce(local_processed_tensor, reduction="sum")


            if accelerator.is_main_process:
                log_info = {k: torch.mean(torch.stack(v)).item() for k, v in info.items()}

                log_info.update({
                    "newly_succeeded_in_step": global_newly_succeeded_sum.item(),
                    "succeeded_in_total": global_succeeded_prompts + global_succeeded_sum.item(),
                    'processed_in_total': global_processed_tensor.item(),
                    "remaining_in_total": global_total_prompts - global_processed_tensor.item(),
                })
                wandb.log(log_info, step=global_step)
            
            # 🚀 Early stopping logic
            # if global_newly_succeeded_sum.item() == 0:
            #     early_stopping_counter += 1
            # else:
            #     early_stopping_counter = 0 # Reset if there's progress
            
            # if early_stopping_counter >= config.train.early_stopping_patience:
            #     accelerator.print(f"No progress for {config.train.early_stopping_patience} consecutive steps. Stopping optimization for this batch.")
            #     break
            global_step += 1
            
            # 🚀 动态B_target更新逻辑
            # 1. 增加所有样本的尝试次数
            for item in B_target:
                prompt_key = _prompt_key(item)
                sample_attempts[prompt_key] += 1
            
            # 2. 找出需要移除的样本（已成功或尝试次数超过限制）
            indices_to_remove = []
            for i, item in enumerate(B_target):
                prompt_key = _prompt_key(item)
                if succeeded_mask[i] or sample_attempts[prompt_key] >= max_attempts_per_sample:
                    indices_to_remove.append(i)

            add_num = 0

            # Save one final output for every sample before it leaves the active pool.
            # Paper-style ASR/HS is computed over all final images, including failures.
            for i in indices_to_remove:
                item = B_target[i]
                prompt_key = _prompt_key(item)
                if prompt_key in finalized_prompt_keys:
                    continue
                attempts = sample_attempts[prompt_key]
                success = bool(succeeded_mask[i].detach().cpu().item())
                final_reason = "success" if success else "max_attempts"
                _write_final_record(
                    config.save_dir,
                    accelerator.process_index,
                    global_step,
                    i,
                    item,
                    merged_images[i],
                    rewards_tensor[i].detach().cpu().item(),
                    attempts,
                    success,
                    final_reason,
                    success_threshold,
                )
                finalized_prompt_keys.add(prompt_key)
            
            # 3. 从后往前移除样本，避免索引变化
            for i in sorted(indices_to_remove, reverse=True):
                # 4. 尝试添加新样本
                try:
                    new_item = next(pending_iterator)
                except StopIteration:
                    # 如果没有更多数据，不添加新样本，也不移除旧样本
                    accelerator.print(f"进程 {accelerator.process_index}: 所有样本已处理完毕或达到最大尝试次数。")
                    break

                add_num += 1

                removed_item = B_target.pop(i)
                prompt_key = _prompt_key(removed_item)
                sample_attempts.pop(prompt_key, None)
                sample_consecutive_successes.pop(prompt_key, None)
                global_processed_prompts += 1

                B_target.insert(i, new_item)
                new_prompt_key = _prompt_key(new_item)
                sample_attempts[new_prompt_key] = 0
                sample_consecutive_successes[new_prompt_key] = 0
                

            
            # 5. 更新成功掩码以匹配新的B_target
            if indices_to_remove:
                succeeded_mask = torch.zeros(len(B_target), dtype=torch.bool, device=accelerator.device)
                # 重新计算当前批次中已成功的样本
                for i, item in enumerate(B_target):
                    prompt_key = _prompt_key(item)
                    # 这里可以添加逻辑检查该样本是否之前已经成功过
                    # 但为了简化，我们假设新添加的样本都未成功
                
                # 如果B_target为空，跳出内部循环
                # if not B_target:
                #     accelerator.print(f"进程 {accelerator.process_index}: 所有样本已处理完毕或达到最大尝试次数。")
                #     break

            global_succeeded_prompts += global_succeeded_sum.item()
            pbar.update(add_num)

            # 🚀 【核心修正】在这里加入全局同步来判断是否可以提前退出
            # 1. 获取本地状态
            status_flag = (add_num == 0) and (len(indices_to_remove) == len(B_target))

            # 2. 将本地状态（一个布尔值）转换为张量，以便进行同步
            local_status_tensor = torch.tensor([status_flag], dtype=torch.int, device=accelerator.device)

            # 3. 使用 gather 收集所有进程的状态
            # gather 会返回一个张量，例如在8卡环境下，它可能是 [1, 0, 1, 1, 0, 1, 1, 1]
            # 其中 1 代表 True, 0 代表 False
            global_status_tensor = accelerator.gather(local_status_tensor)

            # 4. 基于全局状态做决策
            # 全局同步判断是否可以提前退出内部循环
            if global_status_tensor.all():
                accelerator.print(f"所有进程的批次均已攻克！ 同步退出。")
                break # ✅ 这个 break 是安全的

    accelerator.wait_for_everyone()
    logger.info("所有进程均已完成任务！程序结束。")
        
if __name__ == "__main__":
    app.run(main)
