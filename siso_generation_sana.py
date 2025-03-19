#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import copy
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, Gemma2Model

import diffusers
from diffusers import (
    AutoencoderDC,
    FlowMatchEulerDiscreteScheduler,
    SanaPipeline,
    SanaTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module

from local_pipelines.pipeline_sana_with_grads import SanaPipelineWithGrads
from utils.enums import CriterionType
import utils.classifier_utils as classifier_utils
import utils.general_utils as general_utils
import utils.ir_features_utils as ir_features_utils
import utils.alpha_clip_utils as alpha_clip_utils
import utils.detection_utils as detection_utils
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torchvision.transforms.functional as TF
from PIL import Image

pl.seed_everything(35)

logger = get_logger(__name__)

if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


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
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many times to repeat the training data.",
    )

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    # parser.add_argument(
    #     "--instance_prompt",
    #     type=str,
    #     default=None,
    #     required=True,
    #     help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    # )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=300,
        help="Maximum sequence length to use with with the Gemma model",
    )
    parser.add_argument(
        "--complex_human_instruction",
        type=str,
        default=None,
        help="Instructions for complex human attention: https://github.com/NVlabs/Sana/blob/main/configs/sana_app_config/Sana_1600M_app.yaml#L55.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sana-dreambooth-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
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
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
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
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=(
            'We default to the "none" weighting scheme for uniform sampling and uniform loss'
        ),
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_decouple",
        type=bool,
        default=True,
        help="Use AdamW style decoupled weight decay",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-04,
        help="Weight decay to use for unet params",
    )
    parser.add_argument(
        "--adam_weight_decay_text_encoder",
        type=float,
        default=1e-03,
        help="Weight decay to use for text_encoder",
    )

    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma seperated. E.g. - "to_k,to_q,to_v" will result in lora training of attention layers only'
        ),
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
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
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Whether to offload the VAE and the text encoder to CPU when they are not used.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--enable_vae_tiling",
        action="store_true",
        help="Enabla vae tiling in log validation",
    )
    parser.add_argument(
        "--enable_npu_flash_attention",
        action="store_true",
        help="Enabla Flash Attention for NPU",
    )
    # added args
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--criterion_type", type=str, required=True)
    parser.add_argument("--classifier_model_name", type=str, required=True)
    parser.add_argument("--num_ref_images", type=int, required=False)
    parser.add_argument("--subject_image_path_or_url", type=str, required=False)
    parser.add_argument("--subject_class", type=str, required=False, default=None)
    parser.add_argument("--ir_features_weight", type=float, default=1.0)
    parser.add_argument("--dino_features_weight", type=float, default=1.0)
    parser.add_argument("--early_stopping_max_count", type=int, default=5)
    parser.add_argument("--early_stopping_threshold_percentage", type=int, default=5)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--crop_generated_image", action="store_true")
    parser.add_argument("--use_init_image", action="store_true")
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--save_weights", action="store_true")
    parser.add_argument("--weights_output_dir", type=str, default=None)
    parser.add_argument("--log_every_epoch", action="store_true")
    parser.add_argument("--save_loss_threshold", type=float, default=None)
    parser.add_argument("--num_grad_steps", type=int, default=1)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main(args):
    print("loading subject image...")
    subject_image = general_utils.load_image(args.subject_image_path_or_url)
    subject_image_arr = general_utils.image_to_tensor(subject_image)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        revision=args.revision,
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder = Gemma2Model.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    vae = AutoencoderDC.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = SanaTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
    )

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # VAE should always be kept in fp32 for SANA (?)
    vae.to(dtype=torch.float32)
    transformer.to(accelerator.device, dtype=weight_dtype)
    # because Gemma2 is particularly suited for bfloat16.
    text_encoder.to(dtype=torch.bfloat16)

    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            for block in transformer.transformer_blocks:
                block.attn2.set_use_npu_flash_attention(True)
        else:
            raise ValueError(
                "npu flash attention requires torch_npu extensions and is supported only on npu device "
            )

    # Initialize a text encoding pipeline and keep it to CPU for now.
    text_encoding_pipeline = SanaPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = ["to_k", "to_q", "to_v"]

    # now we will add new LoRA weights the transformer layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            SanaPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = SanaPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(
            transformer_, transformer_state_dict, adapter_name="default"
        )
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
        if args.mixed_precision == "fp16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16" or args.mixed_precision == "bf16":
        # if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(
        filter(lambda p: p.requires_grad, transformer.parameters())
    )

    # Optimization parameters
    transformer_parameters_with_lr = {
        "params": transformer_lora_parameters,
        "lr": args.learning_rate,
    }
    params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.optimizer.lower() == "adamw":
        optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError(
                "To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`"
            )

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # def compute_text_embeddings(prompt, text_encoding_pipeline):
    #     text_encoding_pipeline = text_encoding_pipeline.to(accelerator.device)
    #     with torch.no_grad():
    #         prompt_embeds, prompt_attention_mask, _, _ = (
    #             text_encoding_pipeline.encode_prompt(
    #                 prompt,
    #                 max_sequence_length=args.max_sequence_length,
    #                 complex_human_instruction=args.complex_human_instruction,
    #             )
    #         )
    #     if args.offload:
    #         text_encoding_pipeline = text_encoding_pipeline.to("cpu")
    #     prompt_embeds = prompt_embeds.to(transformer.dtype)
    #     return prompt_embeds, prompt_attention_mask

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    # if not train_dataset.custom_instance_prompts:
    #     instance_prompt_hidden_states, instance_prompt_attention_mask = (
    #         compute_text_embeddings(args.instance_prompt, text_encoding_pipeline)
    #     )

    # Handle class prompt for prior-preservation.
    # if args.with_prior_preservation:
    #     class_prompt_hidden_states, class_prompt_attention_mask = (
    #         compute_text_embeddings(args.class_prompt, text_encoding_pipeline)
    #     )

    # Clear the memory here
    # if not train_dataset.custom_instance_prompts:
    #     del text_encoder, tokenizer
    #     free_memory()

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.
    # if not train_dataset.custom_instance_prompts:
    #     prompt_embeds = instance_prompt_hidden_states
    #     prompt_attention_mask = instance_prompt_attention_mask
    #     if args.with_prior_preservation:
    #         prompt_embeds = torch.cat(
    #             [prompt_embeds, class_prompt_hidden_states], dim=0
    #         )
    #         prompt_attention_mask = torch.cat(
    #             [prompt_attention_mask, class_prompt_attention_mask], dim=0
    #         )

    vae_config_scaling_factor = vae.config.scaling_factor
    # if args.cache_latents:
    #     latents_cache = []
    #     vae = vae.to(accelerator.device)
    #     for batch in tqdm(train_dataloader, desc="Caching latents"):
    #         with torch.no_grad():
    #             batch["pixel_values"] = batch["pixel_values"].to(
    #                 accelerator.device, non_blocking=True, dtype=vae.dtype
    #             )
    #             latents_cache.append(vae.encode(batch["pixel_values"]).latent)

    #     if args.validation_prompt is None:
    #         del vae
    #         free_memory()

    # Scheduler and math around the number of training steps.
    # overrode_max_train_steps = False
    # num_update_steps_per_epoch = math.ceil(
    #     len(train_dataloader) / args.gradient_accumulation_steps
    # )
    # if args.max_train_steps is None:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    #     overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.num_train_epochs,
        # num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    # transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #     transformer, optimizer, train_dataloader, lr_scheduler
    # )
    transformer, optimizer, lr_scheduler = accelerator.prepare(
        transformer, optimizer, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # num_update_steps_per_epoch = math.ceil(
    #     len(train_dataloader) / args.gradient_accumulation_steps
    # )
    # if overrode_max_train_steps:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # # Afterwards we recalculate our number of training epochs
    # args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "dreambooth-sana-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    # total_batch_size = (
    #     args.train_batch_size
    #     * accelerator.num_processes
    #     * args.gradient_accumulation_steps
    # )

    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    # logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    # logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    # logger.info(
    #     f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    # )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        accelerator.print(f"Resuming from checkpoint {args.resume_from_checkpoint}")
    # if args.resume_from_checkpoint:
    #     if args.resume_from_checkpoint != "latest":
    #         path = os.path.basename(args.resume_from_checkpoint)
    #     else:
    #         # Get the mos recent checkpoint
    #         dirs = os.listdir(args.output_dir)
    #         dirs = [d for d in dirs if d.startswith("checkpoint")]
    #         dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    #         path = dirs[-1] if len(dirs) > 0 else None

    #     if path is None:
    #         accelerator.print(
    #             f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
    #         )
    #         args.resume_from_checkpoint = None
    #         initial_global_step = 0
    #     else:
    #         accelerator.print(f"Resuming from checkpoint {path}")
    #         accelerator.load_state(os.path.join(args.output_dir, path))
    #         global_step = int(path.split("-")[1])

    #         initial_global_step = global_step
    #         first_epoch = global_step // num_update_steps_per_epoch

    # else:
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.num_train_epochs),
        # range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
    #     sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
    #     schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
    #     timesteps = timesteps.to(accelerator.device)
    #     step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    #     sigma = sigmas[step_indices].flatten()
    #     while len(sigma.shape) < n_dim:
    #         sigma = sigma.unsqueeze(-1)
    #     return sigma

    # get criterion
    if (
        args.criterion_type == CriterionType.ir_features.value
        or args.criterion_type == CriterionType.ir_dino_ensemble.value
    ):
        ir_criterion = general_utils.get_criterion(CriterionType.ir_features.value)
    if (
        args.criterion_type == CriterionType.classifier_features.value
        or args.criterion_type == CriterionType.ir_dino_ensemble.value
    ):
        dino_criterion = general_utils.get_criterion(
            CriterionType.classifier_features.value
        )
    if args.criterion_type == CriterionType.alpha_clip.value:
        alpha_clip_criterion = general_utils.get_criterion(
            CriterionType.alpha_clip.value
        )

    # get model
    if (
        args.criterion_type == CriterionType.ir_features.value
        or args.criterion_type == CriterionType.ir_dino_ensemble.value
    ):
        ir_feature_extractor, feature_extractor_transforms = (
            ir_features_utils.get_ir_model_and_transforms(
                "/cortex/users/yairshp/pretrained_models/IR_features/ir_features.pth",
                device=accelerator.device,
            )
        )  # TODO add argument for model path
        ir_feature_extractor.eval()
        ir_feature_extractor = ir_feature_extractor.to(accelerator.device)
    if (
        args.criterion_type == CriterionType.classifier_features.value
        or args.criterion_type == CriterionType.ir_dino_ensemble.value
    ):
        classifier, transforms_configs = (
            classifier_utils.get_model_and_transforms_configs(
                args.classifier_model_name
            )
        )
    if args.criterion_type == CriterionType.alpha_clip.value:
        alpha_clip_model = alpha_clip_utils.load_alpha_clip()

    # get reference features
    if (
        args.criterion_type == CriterionType.ir_features.value
        or args.criterion_type == CriterionType.ir_dino_ensemble.value
    ):
        with torch.no_grad():
            ir_subject_image_features = ir_features_utils.get_ir_features(
                ir_feature_extractor, feature_extractor_transforms, subject_image_arr
            )
    if (
        args.criterion_type == "classifier_features"
        or args.criterion_type == CriterionType.ir_dino_ensemble.value
    ):
        dino_subject_image_input = classifier_utils.prepare_for_classifier(
            subject_image_arr, transforms_configs
        ).cuda()
        with torch.no_grad():
            dino_subject_image_features = classifier_utils.get_classifier_features(
                classifier, dino_subject_image_input
            )

    pipeline = SanaPipelineWithGrads.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=accelerator.unwrap_model(transformer),
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    ).to(accelerator.device)

    ir_losses = []
    dino_losses = []
    total_losses = []
    best_loss = float("inf")
    best_image = None
    early_stopping_count = 0
    global_step = 0

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        generator = torch.Generator(device=accelerator.device)
        generator.manual_seed(args.seed)
        models_to_accumulate = [transformer]
        with accelerator.accumulate(models_to_accumulate):
            image = (
                pipeline(
                    prompt=args.prompt,
                    num_inference_steps=args.num_inference_steps,
                    output_type="pt",
                    height=args.resolution,
                    width=args.resolution,
                    generator=generator,
                    # guidance_scale=0.0,
                    max_sequence_length=256,
                    num_grad_steps=args.num_grad_steps,
                )
                .images[0]
                .unsqueeze(0)
                .float()
            )
            image_out = image  # save for visaualization

            loss = 0
            if (
                args.criterion_type == CriterionType.ir_features.value
                or args.criterion_type == CriterionType.ir_dino_ensemble.value
            ):
                ir_loss = ir_criterion(
                    ir_feature_extractor,
                    feature_extractor_transforms,
                    image,
                    ir_subject_image_features,
                )
                ir_losses.append(ir_loss.detach().item())
                loss += args.ir_features_weight * ir_loss
            if (
                args.criterion_type == CriterionType.classifier_features.value
                or args.criterion_type == CriterionType.ir_dino_ensemble.value
            ):
                dino_loss = dino_criterion(
                    classifier,
                    transforms_configs,
                    image,
                    dino_subject_image_features,
                )
                dino_losses.append(dino_loss.detach().item())
                loss += args.dino_features_weight * dino_loss

            total_losses += [loss.detach().item()]

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = transformer.parameters()
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if loss < best_loss:
                image_out = general_utils.numpy_to_pil(
                    image_out.detach().permute(0, 2, 3, 1).cpu().numpy()
                )[0]
                best_image = image_out

                if args.save_weights and accelerator.is_main_process:
                    best_transformer = unwrap_model(transformer)
                    best_transformer_lora_layers = get_peft_model_state_dict(
                        transformer
                    )

            if loss < best_loss * (1 + args.early_stopping_threshold_percentage / 100):
                best_loss = loss
                early_stopping_count = 0  # TODO check if zeroing is best option
            else:
                print(
                    f"Early stopping count: {early_stopping_count}. Best loss: {best_loss}, Current loss: {loss}"
                )
                early_stopping_count += 1

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if args.log_every_epoch and accelerator.is_main_process:
                if type(image_out) != Image.Image:
                    image_out = general_utils.numpy_to_pil(
                        image_out.detach().permute(0, 2, 3, 1).cpu().numpy()
                    )[0]
                image_out.save(f"{args.output_dir}/epoch_{epoch}.png")

                general_utils.log_all_losses(
                    {
                        "total_losses": total_losses,
                        **(
                            {"ir_losses": ir_losses, "dino_losses": dino_losses}
                            if args.criterion_type
                            == CriterionType.ir_dino_ensemble.value
                            else {}
                        ),
                    },
                    args.output_dir,
                )

            if early_stopping_count >= args.early_stopping_max_count:
                print(f"Early stopping at epoch {epoch}")
                break

    if args.save_loss_threshold is None or best_loss < args.save_loss_threshold:
        # save losses array to disk as an npy file
        np.save(f"{args.output_dir}/total_losses.npy", np.array(total_losses))
        if args.criterion_type == CriterionType.ir_dino_ensemble.value:
            np.save(f"{args.output_dir}/ir_losses.npy", np.array(ir_losses))
            np.save(f"{args.output_dir}/dino_losses.npy", np.array(dino_losses))

        # save best image
        best_image.save(f"{args.output_dir}/result.png")

        if not args.log_every_epoch:
            general_utils.log_all_losses(
                {
                    "total_losses": total_losses,
                    **(
                        {"ir_losses": ir_losses, "dino_losses": dino_losses}
                        if args.criterion_type == CriterionType.ir_dino_ensemble.value
                        else {}
                    ),
                },
                args.output_dir,
            )

    del pipeline
    torch.cuda.empty_cache()

    accelerator.end_training()

    if args.save_weights:
        Path(os.path.join(args.weights_output_dir, "weights", "lora")).mkdir(
            parents=True, exist_ok=True
        )
        SanaPipeline.save_lora_weights(
            save_directory=os.path.join(args.weights_output_dir, "weights", "lora"),
            transformer_lora_layers=best_transformer_lora_layers,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
