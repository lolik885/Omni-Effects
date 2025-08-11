import os
import gc
import sys
import math
import json
import wandb
import shutil
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from typing import List, Optional, Tuple, Union, Any, Dict
from safetensors.torch import load_file

import torch
from torchvision.transforms.functional import crop, center_crop, resize

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed

import transformers
from transformers import T5EncoderModel, T5Tokenizer
from transformers.models.mixtral.modeling_mixtral import (
    load_balancing_loss_func
)

import diffusers
from diffusers import (
    AutoencoderKLCogVideoX, 
    CogVideoXDPMScheduler, 
    CogVideoXImageToVideoPipeline
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft, export_to_video, load_image
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling
)

from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

from .utils import unwrap_model, get_memory_statistics, free_memory, print_memory, resize_numpy_image_long
# from cogvideox.data.dataset_video import VideoDataset, collate_fn
from cogvideox.data.dataset_video_mask import VideoDataset, collate_fn
from cogvideox.models.omnieffects.attention_processor import CogVideoXAttnLoraProcessor2_0
from cogvideox.models.omnieffects.cogvideox_transformer_3d import CogVideoXTransformer3DModel


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")


class Trainer:
    def __init__(self, args):
        self.args = args
        if self.args.report_to == "wandb" and self.args.hub_token is not None:
            raise ValueError(
                "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
                " Please use `huggingface-cli login` to authenticate with the Hub."
            )

        self._init_distributed()
        self._init_logging()
        self._init_directories()
        self._init_weight_dtype()
        self._init_models()
        self._init_noise_scheduler()
        self._init_sampler(alpha=self.args.sampler_alpha)

    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)

        accelerator_project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=str(logging_dir))

        accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
            project_config=accelerator_project_config
        )

        # Disable AMP for MPS. A technique for accelerating machine learning computations on iOS and macOS devices.
        if torch.backends.mps.is_available():
            logger.info("MPS is enabled. Disabling AMP.")
            accelerator.native_amp = False

        self.accelerator = accelerator

        # If passed along, set the training seed now.
        if self.args.seed is not None:
            set_seed(self.args.seed)

    def _init_logging(self):
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            # DEBUG, INFO, WARNING, ERROR, CRITICAL
            level=logging.INFO,
        )

        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        logger.info("Initialized Trainer")
        logger.info(f"Accelerator state: \n{self.accelerator.state}", main_process_only=False)

    def _init_directories(self):
        # Handle the repository creation
        if self.accelerator.is_main_process:
            if self.args.output_dir is not None:
                os.makedirs(self.args.output_dir, exist_ok=True)

    def _init_weight_dtype(self):
        # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if self.accelerator.state.deepspeed_plugin:
            # DeepSpeed is handling precision, use what's in the DeepSpeed config
            if (
                "fp16" in self.accelerator.state.deepspeed_plugin.deepspeed_config
                and self.accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
            ):
                weight_dtype = torch.float16
            if (
                "bf16" in self.accelerator.state.deepspeed_plugin.deepspeed_config
                and self.accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
            ):
                weight_dtype = torch.bfloat16
        else:
            if self.accelerator.mixed_precision == "fp16":
                weight_dtype = torch.float16
            elif self.accelerator.mixed_precision == "bf16":
                weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype
    
    # 3D VAE
    def _load_vae(self):
        vae = AutoencoderKLCogVideoX.from_pretrained(
            self.args.pretrained_model_name_or_path, 
            subfolder="vae", revision=self.args.revision, variant=self.args.variant
        )
        return vae

    # Transformer
    def _load_transformer(self):
        # CogVideoX-2b weights are stored in float16
        # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
        load_dtype = torch.bfloat16 if "5b" in self.args.pretrained_model_name_or_path.lower() else torch.float16
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=load_dtype,
            revision=self.args.revision,
            variant=self.args.variant,
        )
        return transformer

    # Text encoder
    def _load_text_encoder(self):
        # Prepare models and scheduler
        tokenizer = T5Tokenizer.from_pretrained(
            self.args.pretrained_model_name_or_path, 
            subfolder="tokenizer", revision=self.args.revision
        )

        text_encoder = T5EncoderModel.from_pretrained(
            self.args.pretrained_model_name_or_path, 
            subfolder="text_encoder", revision=self.args.revision
        )
        return tokenizer, text_encoder

    def _init_models(self):
        logger.info("Initializing models")

        self.vae = self._load_vae()
        self.transformer = self._load_transformer()
        self.tokenizer, self.text_encoder = self._load_text_encoder()
        
        if self.args.enable_slicing:
            self.vae.enable_slicing()
        if self.args.enable_tiling:
            self.vae.enable_tiling()

        # self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=torch.float32)
        self.transformer.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

    def _init_noise_scheduler(self):
        logger.info("Initializing noise scheduler")

        noise_scheduler = CogVideoXDPMScheduler.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="scheduler"
        )

        self.noise_scheduler = noise_scheduler

    def _init_sampler(self, alpha=0):
        nums = torch.arange(0, 1000, dtype=torch.float32)
        weights_exp = torch.exp(alpha * nums)
        self.probs_exp = weights_exp / weights_exp.sum()
        self.probs_exp.to(device=self.accelerator.device)

    def prepare_dataset(self):
        logger.info("Initializing dataset and dataloader")

        # Prepare dataset and dataloader.
        train_dataset = VideoDataset(
            [self.args.train_data_meta],
            video_sample_size=[self.args.video_sample_height, self.args.video_sample_width],
            video_sample_n_frames=self.args.video_sample_n_frames,
            video_length_drop_start=0.0,
            video_length_drop_end=1.0,
            random_crop_prob=self.args.random_crop_prob,
            num_cond=self.args.num_cond if self.args.cond_lora else 0
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader

    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")

        if torch.backends.mps.is_available() and self.args.mixed_precision == "bf16":
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)

        self.__prepare_saving_loading_hooks()

        if self.args.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        if self.args.mha_lora:
            self.transformer_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.network_alpha,
                init_lora_weights=True,
                target_modules=['to_k', 'to_q', 'to_v', 'to_out.0']
            )
            self.transformer.add_adapter(self.transformer_lora_config)
        
        if self.args.moe_lora:
            self.transformer.add_moe_lora(
                rank=self.args.rank,
                network_alpha=self.args.network_alpha,
                num_experts=self.args.num_experts,
                top_k=self.args.top_k,
                device=self.accelerator.device, 
                dtype=self.weight_dtype,
            )

        print(type(self.args.cond_lora), self.args.cond_lora)

        if self.args.cond_lora:
            attn_processor_type = CogVideoXAttnLoraProcessor2_0

            dim = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim

            ranks = [self.args.rank] * self.args.num_cond
            lora_weights = [self.args.lora_weight] * self.args.num_cond
            network_alphas = [self.args.network_alpha] * self.args.num_cond

            if self.args.base_lora:
                ranks = [self.args.base_lora_rank] + ranks
                lora_weights = [self.args.base_lora_weight] + lora_weights
                network_alphas = [self.args.base_network_alpha] + network_alphas

            attn_processors = {}
            for key, value in self.transformer.attn_processors.items():
                block_idx = int(key.split(".")[1])

                if block_idx in list(range(self.transformer.config.num_layers)):
                    attn_processor = attn_processor_type(
                        dim=dim, 
                        ranks=ranks,
                        lora_weights=lora_weights,
                        network_alphas=network_alphas,
                        device=self.accelerator.device, 
                        dtype=self.weight_dtype,
                        cond_width=self.args.video_sample_width,
                        cond_height=self.args.video_sample_height,
                        n_loras=self.args.num_cond,
                        cond_lora=self.args.cond_lora,
                        base_lora=self.args.base_lora,
                        text_visual_attention=self.args.text_visual_attention,
                        m2v_mask=self.args.m2v_mask,
                        full_attention=self.args.full_attention
                    )
                    for idx in range(1, self.args.num_cond):
                        attn_processor.cond_q_loras[idx].down.weight    = attn_processor.cond_q_loras[0].down.weight
                        attn_processor.cond_q_loras[idx].up.weight      = attn_processor.cond_q_loras[0].up.weight
                        attn_processor.cond_k_loras[idx].down.weight    = attn_processor.cond_k_loras[0].down.weight
                        attn_processor.cond_k_loras[idx].up.weight      = attn_processor.cond_k_loras[0].up.weight
                        attn_processor.cond_v_loras[idx].down.weight    = attn_processor.cond_v_loras[0].down.weight
                        attn_processor.cond_v_loras[idx].up.weight      = attn_processor.cond_v_loras[0].up.weight
                        attn_processor.cond_proj_loras[idx].down.weight = attn_processor.cond_proj_loras[0].down.weight
                        attn_processor.cond_proj_loras[idx].up.weight   = attn_processor.cond_proj_loras[0].up.weight

                        attn_processor.cond_q_loras[idx].down.weight.data    = attn_processor.cond_q_loras[0].down.weight.data
                        attn_processor.cond_q_loras[idx].up.weight.data      = attn_processor.cond_q_loras[0].up.weight.data
                        attn_processor.cond_k_loras[idx].down.weight.data    = attn_processor.cond_k_loras[0].down.weight.data
                        attn_processor.cond_k_loras[idx].up.weight.data      = attn_processor.cond_k_loras[0].up.weight.data
                        attn_processor.cond_v_loras[idx].down.weight.data    = attn_processor.cond_v_loras[0].down.weight.data
                        attn_processor.cond_v_loras[idx].up.weight.data      = attn_processor.cond_v_loras[0].up.weight.data
                        attn_processor.cond_proj_loras[idx].down.weight.data = attn_processor.cond_proj_loras[0].down.weight.data
                        attn_processor.cond_proj_loras[idx].up.weight.data   = attn_processor.cond_proj_loras[0].up.weight.data
                        
                    for param in attn_processor.parameters():
                        param.requires_grad_(True)
                    attn_processors[key] = attn_processor
                else:
                    attn_processors[key] = value

            self.transformer.set_attn_processor(attn_processors)

            self.transformer.n_loras = self.args.num_cond

            self.transformer.add_cond_linear(
                self.args.video_sample_height // 8, 
                self.args.video_sample_width // 8,
                cond_linear=self.args.cond_linear
            )

        self.transformer.to(self.accelerator.device, dtype=self.weight_dtype)

    def prepare_optimizer(self):
        logger.info("Initializing optimizer and lr scheduler")

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if self.args.scale_lr:
            self.args.learning_rate = (
                self.args.learning_rate * \
                self.args.gradient_accumulation_steps * \
                self.args.train_batch_size * \
                self.accelerator.num_processes
            )

        # Make sure the trainable params are in float32.
        if self.args.mixed_precision == "fp16":
            models = [self.transformer]
            # only upcast trainable parameters into fp32
            cast_training_params(models, dtype=torch.float32)

        # Initialize the optimizer
        if self.args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        transformer_lora_parameters = list(filter(lambda p: p.requires_grad, self.transformer.parameters()))

        params_to_optimize = [
            {
                'params': transformer_lora_parameters, 
                "lr": self.args.learning_rate
            }
        ]

        optimizer = optimizer_cls(
            params_to_optimize,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )

        # Scheduler and math around the number of training steps.
        # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
        if self.args.max_train_steps is None:
            len_train_dataloader_after_sharding = math.ceil(len(self.train_dataloader) / self.accelerator.num_processes)
            num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / self.args.gradient_accumulation_steps)
            num_training_steps_for_scheduler = (
                self.args.num_train_epochs * num_update_steps_per_epoch * self.accelerator.num_processes
            )
        else:
            num_training_steps_for_scheduler = self.args.max_train_steps * self.accelerator.num_processes

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=num_training_steps_for_scheduler,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_training_steps_for_scheduler = num_training_steps_for_scheduler

    def prepare_for_training(self):
        # Prepare everything with our `accelerator`.
        self.transformer, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.transformer, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
            if self.num_training_steps_for_scheduler != self.args.max_train_steps * self.accelerator.num_processes:
                logger.warning(
                    f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                    f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                    f"This inconsistency may result in the learning rate scheduler not functioning properly."
                )
        # Afterwards we recalculate our number of training epochs
        self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)
        self.num_update_steps_per_epoch = num_update_steps_per_epoch

    def prepare_trackers(self):
        logger.info("Initializing trackers")

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            tracker_config = dict(vars(self.args))
            self.accelerator.init_trackers(self.args.tracker_project_name, config=tracker_config)

            self.accelerator.print("===== Memory before training =====")
            free_memory(self.accelerator.device)
            print_memory(self.accelerator.device)

    def train(self):
        logger.info("Starting training")

        memory_statistics = get_memory_statistics(logger)
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        # Train!
        total_batch_size = self.args.train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(self.train_dataloader)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if not self.args.resume_from_checkpoint:
            initial_global_step = 0
        else:
            if self.args.resume_from_checkpoint != "latest":
                path = self.args.resume_from_checkpoint
            else:
                # Get the mos recent checkpoint
                dirs = os.listdir(self.args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.args.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                if self.args.resume_from_checkpoint == "latest":
                    path = os.path.join(self.args.output_dir, path)
                else:
                    pass
                
                if self.args.mha_lora:
                    lora_weights = load_file(os.path.join(path, "pytorch_lora_weights.safetensors"))
                    self.transformer.load_state_dict(lora_weights, strict=False)
                
                if self.args.moe_lora:
                    moe_state_dict = torch.load(os.path.join(path, "moe_lora.pth"))
                    self.transformer.load_state_dict(moe_state_dict, strict=False)
                
                if self.args.cond_lora:
                    cond_state_dict = torch.load(os.path.join(path, "cond_lora.pth"))
                    self.transformer.load_state_dict(cond_state_dict, strict=False)
                
                self.transformer.to(dtype=torch.bfloat16)
                
                global_step = int(path.split("-")[-1])

                initial_global_step = global_step
                first_epoch = global_step // self.num_update_steps_per_epoch

        progress_bar = tqdm(
            range(0, self.args.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process,
        )

        self.vae_scaling_factor_image = self.vae.config.scaling_factor
        self.vae_scale_factor_spatial = 2 ** (len(self.vae.config.block_out_channels) - 1)

        # For DeepSpeed training
        self.model_config = self.transformer.module.config if hasattr(self.transformer, "module") else self.transformer.config

        if self.args.multi_stream:
            # create extra cuda streams to speedup inpaint vae computation
            vae_stream = torch.cuda.Stream()
        else:
            vae_stream = None

        if self.accelerator.num_processes == 1:
            self.timesteps_starts = [0]
            self.timesteps_ends = [self.noise_scheduler.config.num_train_timesteps]
        elif self.accelerator.num_processes == 2:
            self.timesteps_starts = [0, 900]
            self.timesteps_ends = [self.noise_scheduler.config.num_train_timesteps, self.noise_scheduler.config.num_train_timesteps]
        else:
            patch_size = (self.noise_scheduler.config.num_train_timesteps - 900) // (self.accelerator.num_processes - 2)
            timesteps_starts = np.arange(900, self.noise_scheduler.config.num_train_timesteps, patch_size)
            timesteps_ends = timesteps_starts + patch_size
            if timesteps_ends.shape[0] > self.accelerator.num_processes - 2:
                timesteps_ends[-2] = self.noise_scheduler.config.num_train_timesteps
            else:
                timesteps_ends[-1] = self.noise_scheduler.config.num_train_timesteps
            self.timesteps_starts = np.concatenate([[999, 0], timesteps_starts], axis=0)
            self.timesteps_ends = np.concatenate([[1000, 900], timesteps_ends], axis=0)

        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.transformer.train()

            for step, batch in enumerate(self.train_dataloader):
                models_to_accumulate = [self.transformer]
                with self.accelerator.accumulate(models_to_accumulate):
                    loss = self.compute_loss(batch, vae_stream, epoch, step)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        params_to_clip = self.transformer.parameters()
                        self.accelerator.clip_grad_norm_(params_to_clip, self.args.max_grad_norm)
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                    if self.accelerator.distributed_type == DistributedType.DEEPSPEED or self.accelerator.is_main_process:
                        if global_step % self.args.checkpointing_steps == 0:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if self.args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(self.args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= self.args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - self.args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(self.args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
                            self.accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)

                if global_step >= self.args.max_train_steps:
                    break
            
            if self.accelerator.is_main_process:
                if self.args.validation_prompt is not None and (epoch + 1) % self.args.validation_epochs == 0:
                    self.accelerator.print("===== Memory before validation =====")
                    print_memory(self.accelerator.device)
                    torch.cuda.synchronize(self.accelerator.device)

                    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                        self.args.pretrained_model_name_or_path,
                        transformer=unwrap_model(self.accelerator, self.transformer),
                        scheduler=self.noise_scheduler,
                        revision=self.args.revision,
                        variant=self.args.variant,
                        torch_dtype=self.weight_dtype,
                    )

                    if self.args.enable_slicing:
                        pipe.vae.enable_slicing()
                    if self.args.enable_tiling:
                        pipe.vae.enable_tiling()
                    if self.args.enable_model_cpu_offload:
                        pipe.enable_model_cpu_offload()

                    validation_prompts = self.args.validation_prompt.split(self.args.validation_prompt_separator)
                    validation_images = self.args.validation_images.split(self.args.validation_prompt_separator)
                    for validation_image, validation_prompt in zip(validation_images, validation_prompts):
                        pipeline_args = {
                            "image": load_image(validation_image),
                            "prompt": validation_prompt,
                            "guidance_scale": self.args.guidance_scale,
                            "use_dynamic_cfg": self.args.use_dynamic_cfg,
                            "height": self.args.video_sample_height,
                            "width": self.args.video_sample_width,
                            "num_frames": self.args.video_sample_n_frames,
                            "max_sequence_length": self.model_config.max_text_seq_length,
                        }

                        self.log_validation(
                            pipe=pipe,
                            pipeline_args=pipeline_args,
                            epoch=epoch,
                        )

                    self.accelerator.print("===== Memory after validation =====")
                    print_memory(self.accelerator.device)
                    free_memory(self.accelerator.device)

                    del pipe
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(self.accelerator.device)

            memory_statistics = get_memory_statistics(logger)
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")
        
        # Create the pipeline using the trained modules and save it.
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            transformer = unwrap_model(self.accelerator, self.transformer)
            
            transformer_lora_layers = get_peft_model_state_dict(transformer)

            save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
            
            CogVideoXImageToVideoPipeline.save_lora_weights(
                save_directory=save_path,
                transformer_lora_layers=transformer_lora_layers,
            )

            self.logger.info(f"Saved state to {save_path}")

        free_memory(self.accelerator.device)
        memory_statistics = get_memory_statistics(logger)
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        self.accelerator.end_training()

    def fit(self):
        self.prepare_dataset()
        self.prepare_trainable_parameters()
        self.prepare_optimizer()
        self.prepare_for_training()
        self.prepare_trackers()
        self.train()

    def compute_loss(self, batch, vae_stream, epoch, step):
        # Convert videos and images to latent space
        pixel_values = batch["pixel_values"].to(self.weight_dtype)
        ref_pixel_values = batch["ref_pixel_values"].to(self.weight_dtype)
        if batch["cond_pixel_values"] is not None:
            cond_pixel_values = batch["cond_pixel_values"].to(self.weight_dtype)
        else:
            cond_pixel_values = None
        prompts = batch["captions"]

        bsz = pixel_values.size(0)

        t2v_flag = []
        for _ in range(bsz):
            if np.random.rand() < self.args.noised_image_drop_ratio:
                t2v_flag.append(0)
            else:
                t2v_flag.append(1)
        t2v_flag = torch.from_numpy(np.array(t2v_flag)).to(self.accelerator.device, dtype=self.weight_dtype)

        if self.args.low_vram:
            torch.cuda.empty_cache()
            self.vae.to(self.accelerator.device)
            self.text_encoder.to(self.accelerator.device)
        
        pixel_latents = self.encode_video(
            pixel_values, 
            vae_stream, 
            self.vae.to(self.accelerator.device), 
            self.args.vae_mini_batch, 
            self.weight_dtype
        )
        
        patch_size_t = self.model_config.patch_size_t
        if patch_size_t is not None:
            ncopy = pixel_latents.shape[2] % patch_size_t
            first_frame = pixel_latents[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            pixel_latents = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), pixel_latents], dim=2)
            assert pixel_latents.shape[2] % patch_size_t == 0
        
        pixel_latents = rearrange(pixel_latents, "b c f h w -> b f c h w")

        if cond_pixel_values is not None: 
            if cond_pixel_values.shape[2] > 1:
                cond_latents = torch.tensor([]).to(self.accelerator.device, dtype=self.weight_dtype)
                for i in range(cond_pixel_values.shape[2]):
                    cond_latent = self.encode_video(
                    cond_pixel_values[:,:,i,:,:].unsqueeze(2), 
                    None, 
                    self.vae.to(self.accelerator.device), 
                    self.args.vae_mini_batch, 
                    self.weight_dtype
                    )
                    cond_latents = torch.cat([cond_latents, cond_latent], dim=2)
            else:
                cond_latents = self.encode_video(
                cond_pixel_values, 
                None, 
                self.vae.to(self.accelerator.device), 
                self.args.vae_mini_batch, 
                self.weight_dtype
                )
            cond_latents = rearrange(cond_latents, "b c f h w -> b f c h w")
        else:
            cond_latents = None

        ref_pixel_latents = self.encode_video(
            ref_pixel_values,
            None, 
            self.vae.to(self.accelerator.device), 
            self.args.vae_mini_batch, 
            self.weight_dtype
        )
        ref_pixel_latents = rearrange(ref_pixel_latents, "b c f h w -> b f c h w")
        padding_shape = (pixel_latents.shape[0], pixel_latents.shape[1] - 1, *pixel_latents.shape[2:])
        latent_padding = ref_pixel_latents.new_zeros(padding_shape)
        ref_pixel_latents = torch.cat([ref_pixel_latents, latent_padding], dim=1)
        ref_pixel_latents = t2v_flag[:, None, None, None, None] * ref_pixel_latents

        # wait for latents = vae.encode(pixel_values) to complete
        if vae_stream is not None:
            torch.cuda.current_stream().wait_stream(vae_stream)

        if self.args.low_vram:
            self.vae.to('cpu')
            torch.cuda.empty_cache()
            self.text_encoder.to(self.accelerator.device)

        # encode prompts input: [prompt1, prompt2]
        prompt_embeds = [self.encode_prompt(
            self.tokenizer,
            self.text_encoder,
            prompt,
            num_videos_per_prompt=1,
            device=self.accelerator.device,
            dtype=self.weight_dtype
        ) for prompt in prompts]

        prompt_embeds = torch.cat(prompt_embeds, dim=1)

        if self.args.low_vram:
            self.text_encoder.to('cpu')
            torch.cuda.empty_cache()

        noise = torch.randn_like(pixel_latents, device=self.accelerator.device, dtype=self.weight_dtype)
        
        timesteps = torch.randint(
            self.timesteps_starts[self.accelerator.process_index], 
            self.timesteps_ends[self.accelerator.process_index], 
            (bsz,), 
            device=self.accelerator.device
        )
        timesteps = timesteps.long()

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = self.noise_scheduler.add_noise(pixel_latents, noise, timesteps)
        # Concatenate across channels.
        latent_model_input = torch.cat([noisy_model_input, ref_pixel_latents], dim=2)

        cond_hidden_states = cond_latents

        # Prepare rotary embeds
        image_rotary_emb = (
            self.get_rotary_pos_embed(
                height=self.args.video_sample_height,
                width=self.args.video_sample_width,
                cond_height=self.args.video_sample_height,
                cond_width=self.args.video_sample_width,
                num_frames=pixel_latents.shape[1],
                transformer_config=self.model_config,
                vae_scale_factor_spatial=self.vae_scale_factor_spatial,
                device=self.accelerator.device,
                n_loras=self.args.num_cond if self.args.cond_lora else 0
            )
            if self.model_config.use_rotary_positional_embeddings
            else None
        )
        
        # Predict noise, For CogVideoX1.5 Only.
        ofs_emb = (
            None
            if self.model_config.ofs_embed_dim is None
            else pixel_latents.new_full((1,), fill_value=2.0)
        )
        model_output, all_router_logits = self.transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds,
            cond_hidden_states=cond_hidden_states,
            timestep=timesteps,
            ofs=ofs_emb,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )
        model_pred = self.noise_scheduler.get_velocity(model_output, noisy_model_input, timesteps)
        
        alphas_cumprod = self.noise_scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(model_pred.shape):
            weights = weights.unsqueeze(-1)

        target = pixel_latents

        loss = torch.mean((weights * (model_pred - target) ** 2).reshape(bsz, -1), dim=1)
        loss = loss.mean()

        if self.args.aux_loss:
            aux_loss = load_balancing_loss_func(
                all_router_logits,
                self.args.num_experts,
                self.args.top_k,
            )
            loss += 0.01 * aux_loss

        if self.accelerator.is_main_process and self.args.debug and step % 10 == 0:
            self.vae.to(self.accelerator.device)

            latents = model_pred.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
            latents = 1 / self.vae_scaling_factor_image * latents
            latents = latents.to(self.weight_dtype)

            with torch.no_grad():
                video = self.vae.decode(latents).sample
            video = (video / 2 + 0.5).clamp(0, 1)
            video = video.permute(0, 2, 3, 4, 1)
            video = video.cpu().float().numpy()
            os.makedirs(f"visualization/omnieffects_debug", exist_ok=True)
            export_to_video(video[0], f"visualization/omnieffects_debug/training_epoch{epoch}_step{step}_timestep{timesteps[0].item()}.mp4", fps=8)
            
            self.vae.to('cpu')

        return loss

    def log_validation(self, pipe, pipeline_args, epoch, is_final_validation=False):
        pipe = pipe.to(self.accelerator.device)

        # run inference
        generator = torch.Generator(
            device=self.accelerator.device
        ).manual_seed(self.args.seed) if self.args.seed else None

        videos = []
        for _ in range(self.args.num_validation_videos):
            video = pipe(
                **pipeline_args, 
                generator=generator, 
                output_type="np"
            ).frames[0]
            videos.append(video)

        for tracker in self.accelerator.trackers:
            phase_name = "test" if is_final_validation else "validation"
            if tracker.name == "wandb":
                video_filenames = []
                for i, video in enumerate(videos):
                    prompt = (
                        pipeline_args["prompt"][:25]
                        .replace(" ", "_")
                        .replace(" ", "_")
                        .replace("'", "_")
                        .replace('"', "_")
                        .replace("/", "_")
                    )
                    filename = os.path.join(self.args.output_dir, f"{phase_name}_video_e{epoch}_i{i}_{prompt}.mp4")
                    export_to_video(video, filename, fps=8)#8
                    video_filenames.append(filename)

                tracker.log(
                    {
                        phase_name: [
                            wandb.Video(filename, caption=f"{i}: {pipeline_args['prompt']}")
                            for i, filename in enumerate(video_filenames)
                        ]
                    }
                )

        return videos

    @staticmethod
    def encode_video(pixel_values, vae_stream, vae, vae_mini_batch, weight_dtype):
        with torch.no_grad():
            # This way is quicker when batch grows up
            def _slice_vae(pixel_values):
                bs = vae_mini_batch
                new_pixel_values = []
                for i in range(0, pixel_values.shape[0], bs):
                    pixel_values_bs = pixel_values[i : i + bs]
                    pixel_values_bs = vae.encode(pixel_values_bs).latent_dist
                    pixel_values_bs = pixel_values_bs.sample()
                    new_pixel_values.append(pixel_values_bs)
                return torch.cat(new_pixel_values, dim = 0)
            if vae_stream is not None:
                vae_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(vae_stream):
                    latents = _slice_vae(pixel_values)
            else:
                latents = _slice_vae(pixel_values)

            if (
                hasattr(vae.config, "shift_factor")
                and vae.config.shift_factor
            ):
                latents = (
                    (latents - vae.config.shift_factor)
                    * vae.config.scaling_factor
                )
            else:
                latents = latents * vae.config.scaling_factor
        
        return latents.to(weight_dtype)

    def encode_prompt(
        self,
        tokenizer, 
        text_encoder,
        prompt,
        num_videos_per_prompt=1,
        max_sequence_length=226,
        device=None,
        dtype=None,
        text_input_ids=None
    ):
        with torch.no_grad():
            prompt = [prompt] if isinstance(prompt, str) else prompt
            prompt_embeds = self._get_t5_prompt_embeds(
                tokenizer,
                text_encoder,
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
                text_input_ids=text_input_ids,
            )
            return prompt_embeds

    @staticmethod
    def _get_t5_prompt_embeds(
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        prompt: Union[str, List[str]],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        text_input_ids=None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if tokenizer is not None:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
        else:
            if text_input_ids is None:
                raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

        prompt_embeds = text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    @staticmethod
    def get_rotary_pos_embed(
        height: int,
        width: int,
        cond_height: int,
        cond_width: int,
        num_frames: int,
        transformer_config: Dict,
        vae_scale_factor_spatial: int = 8,
        device: Optional[torch.device] = None,
        n_loras: int = 1
    ):
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if cond_height is not None and cond_width is not None:
            cond_grid_height = cond_height // (vae_scale_factor_spatial * transformer_config.patch_size)
            cond_grid_width = cond_width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (
                num_frames + transformer_config.patch_size_t - 1
            ) // transformer_config.patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        if cond_height is not None:
            cond_freqs_cos, cond_freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=transformer_config.attention_head_dim,
                crops_coords=None,
                grid_size=(cond_grid_height, cond_grid_width),
                temporal_size=1,
                grid_type="slice",
                max_size=(cond_grid_height, cond_grid_width),
                device=device,
            )

            if n_loras > 0:
                for i in range(n_loras):
                    freqs_cos = torch.cat([freqs_cos, cond_freqs_cos], dim=0)
                    freqs_sin = torch.cat([freqs_sin, cond_freqs_sin], dim=0)
                return (freqs_cos, freqs_sin)

        return (
            freqs_cos,
            freqs_sin
        )

    @staticmethod
    def _get_shift_for_sequence_length(
        seq_length: int,
        min_tokens: int = 1024,
        max_tokens: int = 4096,
        min_shift: float = 0.95,
        max_shift: float = 2.05,
    ) -> float:
        # Calculate the shift value for a given sequence length using linear interpolation
        # between min_shift and max_shift based on sequence length.
        m = (max_shift - min_shift) / (max_tokens - min_tokens)  # Calculate slope
        b = min_shift - m * min_tokens  # Calculate y-intercept
        shift = m * seq_length + b  # Apply linear equation y = mx + b
        return shift

    def __prepare_saving_loading_hooks(self):
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                transformer_lora_layers_to_save = None

                for model in models:
                    if isinstance(
                        unwrap_model(self.accelerator, model), 
                        type(unwrap_model(self.accelerator, self.transformer))
                    ):
                        if self.args.cond_lora:
                            cond_lora_save_path = os.path.join(output_dir, "cond_lora.pth")
                            cond_lora_embedding_state_dict = {}
                        if self.args.moe_lora:
                            moe_lora_save_path = os.path.join(output_dir, "moe_lora.pth")
                            moe_lora_embedding_state_dict = {}

                        model = unwrap_model(self.accelerator, model)
                            
                        for name, param in model.state_dict().items():
                            if "lora_moe_block" in name:
                                moe_lora_embedding_state_dict[name] = param
                            if "processor" in name:
                                cond_lora_embedding_state_dict[name] = param
                            elif "cond" in name:
                                cond_lora_embedding_state_dict[name] = param
                            else:
                                pass
                        if self.args.moe_lora:
                            torch.save(moe_lora_embedding_state_dict, moe_lora_save_path)
                        if self.args.cond_lora:
                            torch.save(cond_lora_embedding_state_dict, cond_lora_save_path)
                        if self.args.mha_lora:
                            transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()
                if self.args.mha_lora:
                    CogVideoXImageToVideoPipeline.save_lora_weights(
                        output_dir,
                        transformer_lora_layers=transformer_lora_layers_to_save,
                    )

        def load_model_hook(models, input_dir):
            transformer_ = None

            if not self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()

                    if isinstance(
                        unwrap_model(self.accelerator, model), 
                        type(unwrap_model(self.accelerator, self.transformer))
                    ):
                        transformer_ = model  # noqa: F841
                    else:
                        raise ValueError(f"unexpected save model: {unwrap_model(self.accelerator, model).__class__}")
            else:
                transformer_ = CogVideoXTransformer3DModel.from_pretrained(
                    self.args.pretrained_model_name_or_path, subfolder="transformer"
                )
                if self.args.mha_lora:
                    transformer_.add_adapter(self.transformer_lora_config)
            
            lora_state_dict = CogVideoXImageToVideoPipeline.lora_state_dict(input_dir)

            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
            }
            transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
            incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

            # Make sure the trainable params are in float32. This is again needed since the base models
            # are in `weight_dtype`. More details:
            # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
            if self.args.mixed_precision == "fp16":
                # only upcast trainable parameters (LoRA) into fp32
                cast_training_params([transformer_])

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def tensor_to_pil(self, src_img_tensor):
        """
        Converts a tensor image to a PIL image.

        This function takes an input tensor with the shape (C, H, W) and converts it
        into a PIL Image format. It ensures that the tensor is in the correct data
        type and moves it to CPU if necessary.

        Parameters:
            src_img_tensor (torch.Tensor): Input image tensor with shape (C, H, W),
                where C is the number of channels, H is the height, and W is the width.

        Returns:
            PIL.Image: The converted image in PIL format.
        """

        img = src_img_tensor.clone().detach()
        if img.dtype == torch.bfloat16:
            img = img.to(torch.float32)
        img = img.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = img.astype(np.uint8)
        pil_image = Image.fromarray(img)
        return pil_image
    
    def pixel_values_to_pil(self, pixel_values, frame_index=0):
        if pixel_values.is_cuda:
            pixel_values = pixel_values.clone().cpu()
        pixel_values = (pixel_values + 1.0) / 2.0 * 255.0
        pixel_values = pixel_values.clamp(0, 255).byte()
        frame = pixel_values[:, frame_index]  # [C, H, W]
        frame = frame.permute(1, 2, 0)  # [H, W, C]
        frame_np = frame.numpy()
        image = Image.fromarray(frame_np)
        return image