#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright 2024 The VIP Inc. AIGC team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
from os.path import exists, join
import shutil
from pathlib import Path
import time
import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
import random
import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
)
from diffusers.models.vip.transformer_flux import FluxTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    cast_training_params
)
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.data import FluxMarketDataset, FluxMarketDataset_val, flux_collate_fn, ModelEMA
from diffusers.pipelines.flux.pipeline_flux import calculate_shift
from diffusers.pipelines.vip.flux_image_variation_img2img import FluxImageVariationPipeline
from metrics import MyFID, MyLPIPS, Reconstruction_Metrics, preprocess_path_for_deform_task
logger = get_logger(__name__)
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


def train_RMSNorm_params(model):
    result = []
    for name, param in model.named_parameters():
        if "norm_q" in name or "norm_k" in name:
            param.requires_grad = True
            result.append(name)
        elif "norm_added_q" in name or "norm_added_k" in name:
            param.requires_grad = True
            result.append(name)
    return result


def save_models(output_dir, model):
    if not exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # save state_dict
    iv_state_dict = {}
    for k, v in model.state_dict().items():
        if "time_text_embed" in k or "context_embedder" in k:
            iv_state_dict[k] = v
        elif "norm_q" in k or "norm_k" in k:
            iv_state_dict[k] = v
        elif "norm_added_q" in k or "norm_added_k" in k:
            iv_state_dict[k] = v
        elif "x_embedder" in k:
            iv_state_dict[k] = v
    torch.save(iv_state_dict, join(output_dir, "image_variation.bin"))
    logger.info(f"Saved image variation weight to {output_dir}.")

    # save lora
    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
    FluxPipeline.save_lora_weights(
        output_dir,
        transformer_lora_layers=transformer_lora_layers_to_save,
    )


@torch.no_grad()
def encode_vae_image(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--image_variation_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--transformer_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux_sc_iv",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_file_val",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--prob_uncond",
        type=float,
        default=0.0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0.1.",
    )

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--ema_decay", type=float, default=0.9998)
    parser.add_argument(
        "--resolution",
        type=int,
        default=160,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--task",
        type=str,
        default="market",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=1, help="Batch size (per device) for the validation dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=10,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--use_lpips", action="store_true", help="Whether to use Lpips Loss.")
    parser.add_argument(
        "--w_lpips", type=float, default=0.01, help="lpips weight"
    )
    parser.add_argument(
        "--kpt_thr", type=float, default=0.3, help="kpt_thr"
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--use_time_shift",
        action="store_true",
        help="Whether to use dynamic shifting."
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_init",
        type=str,
        default="gaussian",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--vos_bucket",
        type=str,
        default="public",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        variant=args.variant,
    )
    if args.transformer_model_name_or_path:
        transformer = FluxTransformer2DModel.from_pretrained(
            args.transformer_model_name_or_path, variant=args.variant
        )
    else:
        transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", variant=args.variant
        )

    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.requires_grad_(False)

    # Load lora weights
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank if args.lora_alpha is None else args.lora_alpha,
        init_lora_weights=args.lora_init,
        target_modules=[
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
            "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
            "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2",
            "norm1.linear", "norm1_context.linear",
            "norm.linear", "norm_out.linear",
            "proj_mlp", "proj_out",
        ],
    )
    transformer.add_adapter(transformer_lora_config)
    if args.lora_model_path:
        lora_state_dict = FluxPipeline.lora_state_dict(args.lora_model_path)
        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")

    # Load image variation
    state_dict = None
    if args.image_variation_model_path:
        state_dict = torch.load(args.image_variation_model_path, map_location='cpu')
    transformer._init_image_variation(
        joint_attention_dim=transformer.config.in_channels,
        pooled_projection_dim=None,
        state_dict=state_dict,
        alter_x_embedder=True,
    )

    train_RMSNorm_params(transformer)
    cast_training_params(transformer)

    # make img_ids
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_h = args.resolution // vae_scale_factor
    latent_w = args.width // vae_scale_factor
    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
        args.train_batch_size,
        latent_h // 2,
        latent_w // 2,
        accelerator.device,
        weight_dtype,
    )

    text_ids = latent_image_ids.clone()
    text_ids[..., 0] = 1
    text_ids = text_ids.to(device=accelerator.device, dtype=weight_dtype)

    if args.use_time_shift:
        # compute mu to shifting
        mu = calculate_shift(
            latent_image_ids.shape[0],
            noise_scheduler.config.base_image_seq_len,
            noise_scheduler.config.max_image_seq_len,
            noise_scheduler.config.base_shift,
            noise_scheduler.config.max_shift,
        )
        sigmas = noise_scheduler.time_shift(mu, 1.0, noise_scheduler.sigmas)
        noise_scheduler.sigmas = sigmas
        timesteps = sigmas * noise_scheduler.config.num_train_timesteps
        noise_scheduler.timesteps = timesteps

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimization parameters
    params_to_clip = [p for p in transformer.parameters() if p.requires_grad]
    num_params = sum([a.numel() for a in params_to_clip])
    print(f"Model params: {num_params / 1000 / 1000} M")

    optimizer = optimizer_class(
        params_to_clip,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # Dataset and DataLoaders creation:
    train_dataset = FluxMarketDataset(
        dataset_file=args.dataset_file,
        img_size=args.resolution,
        img_w=args.width,
        prob_uncond=args.prob_uncond,
        vos_bucket=args.vos_bucket,
        kpt_thr=args.kpt_thr
    )

    val_dataset = FluxMarketDataset_val(
        dataset_file=args.dataset_file_val,
        img_size=args.resolution,
        img_w=args.width,
        kpt_thr=args.kpt_thr
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=flux_collate_fn,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
        pin_memory=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    transformer, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # Create EMA for the transformer.
    if args.use_ema:
        ema_transformer = ModelEMA(unwrap_model(transformer), args.ema_decay)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    best_fid, best_lpips, best_ssim, best_epoch = 100.00, 1.0, 0.0000, -1
    # lpips_loss
    if args.use_lpips:
        loss_device = accelerator.device
        lpips_loss = MyLPIPS(loss_device)
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Convert images to latent space
                model_input = encode_vae_image(batch["pixel_values"], vae, weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                control_input = encode_vae_image(batch["conditioning_pixel_values"], vae, weight_dtype)
                noisy_model_input = torch.cat([noisy_model_input, control_input], dim=1)

                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=noisy_model_input.shape[0],
                    num_channels_latents=noisy_model_input.shape[1],
                    height=noisy_model_input.shape[2],
                    width=noisy_model_input.shape[3],
                )

                refer_input = encode_vae_image(batch["reference_pixel_values"], vae, weight_dtype)
                refer_input = refer_input * batch["uncond"].unsqueeze(-1).to(dtype=weight_dtype)
                packed_refer_input = FluxPipeline._pack_latents(
                    refer_input,
                    batch_size=refer_input.shape[0],
                    num_channels_latents=refer_input.shape[1],
                    height=refer_input.shape[2],
                    width=refer_input.shape[3],
                )

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input.detach(),
                    timestep=timesteps,
                    encoder_hidden_states=packed_refer_input.detach(),
                    txt_ids=text_ids.detach(),
                    img_ids=latent_image_ids.detach(),
                    return_dict=False,
                )[0]

                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=int(model_input.shape[2] * vae_scale_factor),
                    width=int(model_input.shape[3] * vae_scale_factor),
                    vae_scale_factor=vae_scale_factor,
                )

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = noise - model_input

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                # loss_mae = torch.mean(
                #     (weighting.float() * (model_pred.float() - target.float())).reshape(target.shape[0], -1),
                #     1,
                # )
                loss = loss.mean()
                # lpips_loss
                if args.use_lpips:
                    model_pred = noisy_model_input[:,:16,:,:] - sigmas * model_pred
                    model_pred = (model_pred / vae.config.scaling_factor) + vae.config.shift_factor
                    model_pred = model_pred.to(dtype=weight_dtype)
                    image = vae.decode(model_pred, return_dict=False)[0]
                    image = (image * 0.5 + 0.5).clamp(0, 1)
                    image = image * 2.0 - 1.0
                    image2 = (batch["pixel_values"] * 0.5 + 0.5).clamp(0, 1)
                    image2 = image2 * 2.0 - 1.0
                    loss_lpips = lpips_loss(image2, image).mean()
                    loss += args.w_lpips * loss_lpips
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_transformer.update()
                progress_bar.update(1)
                global_step += 1

            # logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "lpips": loss_lpips.detach().item()}
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
                
        if accelerator.is_main_process: 
            save_path = os.path.join(args.output_dir, f"checkpoint-{epoch}")
            save_models(save_path, unwrap_model(transformer))
            if args.use_ema:
                ema_transformer.save_model(args.output_dir, weights_name='diffusion_pytorch_model_ema.safetensors')
            logger.info(f"Saved state to {save_path}")
        # val
        accelerator.wait_for_everyone()
        transformer.eval()
        output_image_dir = os.path.join(args.output_dir, f"images/checkpoint-{epoch}")
        os.makedirs(output_image_dir, exist_ok=True)
        pt_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        pipe_pose_trans = FluxImageVariationPipeline(
            scheduler=pt_scheduler,
            vae=vae,
            transformer=transformer,
            alter_x_embedder=True,
            structural_control=True,
            use_karras_sigmas=False,
        ).to(accelerator.device)
        seed = int(random.randrange(4294967294))
        generator = torch.Generator(accelerator.device).manual_seed(seed)
        for step, batch in enumerate(val_dataloader):

            res_img = pipe_pose_trans(
                image=batch["ref_img"],
                reference_image=batch["ref_img"],
                control_image=batch["pose_img"],
                height=args.resolution,
                width=args.width,
                strength=1.0,
                num_inference_steps=30,
                guidance_scale=3.0,
                num_images_per_prompt=len(batch["ref_img"]),
                generator=generator,
                controlnet_conditioning_scale=1.0,
                return_dict=False,
            )
            for i, out_img in enumerate(res_img):
                out_img.save(join(output_image_dir, batch["names"][i] + '.png'))
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            devices = accelerator.device
            fid = MyFID(devices)
            lpips_obj = MyLPIPS(devices)
            rec = Reconstruction_Metrics()

            # distorated_path = './PCDMs_Results/stage3_256_results' 
            distorated_path = output_image_dir
            results_save_path =  distorated_path + '_results.txt'    # save path


            gt_list, distorated_list = preprocess_path_for_deform_task(gt_path, distorated_path)
            print(len(gt_list), len(distorated_list))

            FID = float(fid.calculate_from_disk(distorated_path, real_path, img_size=(img_w,img_h)))
            LPIPS = lpips_obj.calculate_from_disk(distorated_list, gt_list, img_size=(img_w,img_h), sort=False)
            REC = rec.calculate_from_disk(distorated_list, gt_list, distorated_path,  img_size=(img_w,img_h), sort=False, debug=False)
            lpips = float(LPIPS.to('cpu').item())
            ssim = float(REC['ssim_256'][0])
            best_count = 0
            if FID < best_fid:
                best_count += 1
            if lpips < best_lpips:
                best_count += 1
            if ssim > best_ssim:
                best_count += 1
            if best_count >= 3:
                best_fid, best_lpips, best_ssim, best_epoch = FID, lpips, ssim, int(epoch)

            print ("FID: "+str(FID)+"\nLPIPS: "+str(lpips)+"\nSSIM: "+str(ssim))
            with open(args.output_dir + '_results.txt', 'a') as ff:
                FID_rounded = round(FID, 4)
                lpips_rounded = round(lpips, 4)
                ssim_rounded = round(ssim, 4)
                ff.write("Epoch: "+str(epoch)+"\nFID: "+str(FID_rounded)+" LPIPS: "+str(lpips_rounded)+" SSIM: "+str(ssim_rounded)+" "+str(best_count))
                if best_count >= 3:
                    ff.write(" best")
                ff.write("\n")
        else:
            time.sleep(3600)
    # Save the weights
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        save_models(args.output_dir, transformer)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
