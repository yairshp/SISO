# !/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
# limitations under the License.
"""Fine-tuning script for Stable Diffusion XL for text2image with support for LoRA."""

import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    DistributedType,
    ProjectConfiguration,
    set_seed,
)
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict, get_peft_model
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_snr,
)
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from utils.enums import CriterionType
import utils.dino_utils as dino_utils
import utils.general_utils as general_utils
import utils.ir_features_utils as ir_features_utils
import utils.labeler_utils as labeler_utils
import utils.grounding_sam_utils as grounding_sam_utils

import utils.renoise_inversion_utils as renoise_inversion_utils
import pytorch_lightning as pl
from PIL import Image

pl.seed_everything(35)

logger = get_logger(__name__)
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


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
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
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
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
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
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
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
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
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
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
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
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
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
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
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
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
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
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--enable_npu_flash_attention",
        action="store_true",
        help="Whether or not to use npu flash attention.",
    )
    parser.add_argument(
        "--noise_offset", type=float, default=0, help="The scale of noise offset."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--debug_loss",
        action="store_true",
        help="debug loss for each image, if filenames are available in the dataset",
    )
    # added args
    parser.add_argument("--criterion_type", type=str, required=True)
    parser.add_argument(
        "--classifier_model_name", type=str, default="vit_large_patch14_dinov2.lvd142m"
    )
    parser.add_argument("--subject_image_path_or_url", type=str, required=False)
    parser.add_argument("--subject_class", type=str, required=False, default=None)
    parser.add_argument("--input_image_path_or_url", type=str, required=True)
    parser.add_argument("--input_class", type=str, required=False, default=None)
    parser.add_argument(
        "--inversion_type", type=str, choices=["nri", "renoise"], default="nri"
    )
    parser.add_argument("--inversion_max_step", type=float, default=0.75)
    parser.add_argument("--bg_mse_loss_weight", type=float, default=0.0)
    parser.add_argument("--ir_features_weight", type=float, default=1.0)
    parser.add_argument("--dino_features_weight", type=float, default=1.0)
    parser.add_argument("--early_stopping_max_count", type=int, default=5)
    parser.add_argument("--early_stopping_threshold_percentage", type=int, default=5)
    parser.add_argument("--log_every_epoch", action="store_true")
    parser.add_argument("--save_loss_threshold", type=float, default=None)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def get_criterion(criterion_type: str):
    if criterion_type == CriterionType.classifier_features.value:
        return dino_utils.get_dino_features_negative_mean_cos_sim
    if criterion_type == CriterionType.features_multi_model.value:
        return dino_utils.get_features_mean_cos_sim_multi_model
    if criterion_type == CriterionType.intermediate_features.value:
        return dino_utils.get_intermediate_features_mean_cos_sim
    if criterion_type == CriterionType.ir_features.value:
        return ir_features_utils.get_ir_features_negative_mean_cos_sim
    # if criterion_type == CriterionType.alpha_clip.value:
    #     return alpha_clip_utils.get_alpha_clip_features_mean_cos_sim
    raise ValueError(f"Loss type {criterion_type} is not supported.")


def get_inversion_config_and_last_latent(
    args, input_image, pipeline_inversion, pipeline_inference, inversion_prompt
):
    if args.inversion_type == "nri":
        inversion_config = nri_inversion_utils.get_inversion_config()
        last_latent = nri_inversion_utils.get_last_latent(
            inversion_pipeline=pipeline_inversion,
            init_image=input_image,
            inversion_prompt=inversion_prompt,
            inversion_config=inversion_config,
            inversion_hp=[2, 0.1, 0.2],
        )
    elif args.inversion_type == "renoise":
        inversion_config = renoise_inversion_utils.get_inversion_config(
            args.inversion_max_step
        )
        last_latent = renoise_inversion_utils.invert(
            pipe_inversion=pipeline_inversion,
            pipe_inference=pipeline_inference,
            init_image=input_image,
            prompt=inversion_prompt,
            cfg=inversion_config,
        )

    return inversion_config, last_latent


def generate_image(
    inversion_type, pipeline_inference, prompt, last_latent, inv_cfg, generator
):
    if inversion_type == "nri":
        image = (
            pipeline_inference(
                prompt=prompt,
                num_inference_steps=inv_cfg.num_inference_steps,
                negative_propmt="",
                callback_on_step_end=nri_inversion_utils.inference_callback,
                image=last_latent,
                strength=inv_cfg.inversion_max_step,
                denoising_start=1.0 - inv_cfg.inversion_max_step,
                guidance_scale=1.2,
                generator=generator,
                output_type="pt",
            )
            .images[0]
            .unsqueeze(0)
        )
    elif inversion_type == "renoise":
        image = (
            pipeline_inference(
                prompt=prompt,
                num_inference_steps=inv_cfg.num_inference_steps,
                negative_prompt=prompt,
                image=last_latent,
                strength=inv_cfg.inversion_max_step,
                denoising_start=1.0 - inv_cfg.inversion_max_step,
                guidance_scale=1.0,
                output_type="pt",
            )
            .images[0]
            .unsqueeze(0)
        )
    return image


def update_early_stopping_count(
    early_stopping_threshold_percentage, early_stopping_count, best_loss, loss
):
    if loss < best_loss * (1 + early_stopping_threshold_percentage / 100):
        best_loss = loss
        early_stopping_count = 0
    else:
        print(
            f"Early stopping count: {early_stopping_count}. Best loss: {best_loss}, Current loss: {loss}"
        )
        early_stopping_count += 1
    return best_loss, early_stopping_count


# def log_all_losses(
#     total_losses,
#     bg_losses,
#     ir_losses,
#     dino_losses,
#     args,
# ):
#     general_utils.plot_losses(total_losses, f"{args.output_dir}/total_losses.png")
#     if args.bg_mse_loss_weight > 0.0:
#         general_utils.plot_losses(bg_losses, f"{args.output_dir}/bg_losses.png")
#     if args.criterion_type == CriterionType.ir_dino_ensemble.value:
#         general_utils.plot_losses(ir_losses, f"{args.output_dir}/ir_losses.png")
#         general_utils.plot_losses(dino_losses, f"{args.output_dir}/dino_losses.png")


def main(args):
    print("loading images...")
    input_image = general_utils.load_image(args.input_image_path_or_url)
    input_image = general_utils.resize_and_center_crop(
        input_image, (args.resolution, args.resolution)
    )
    input_image_arr = general_utils.image_to_tensor(input_image).cuda()

    subject_image = general_utils.load_image(args.subject_image_path_or_url)
    subject_image_arr = general_utils.image_to_tensor(subject_image)

    print("loading labeler model...")
    labeler_model, labeler_processor = labeler_utils.get_labeler_model_and_processor()

    print("labeling images...")
    if args.input_class is not None:
        input_image_label = args.input_class
    else:
        input_image_label = labeler_utils.get_label(
            input_image, labeler_model, labeler_processor
        )

    if args.subject_class is not None:
        subject_image_label = args.subject_class
    else:
        subject_image_label = labeler_utils.get_label(
            subject_image, labeler_model, labeler_processor
        )

    print("loading Grounding SAM...")
    detection_threshold, detector_model, segmentator_model, segment_processor = (
        grounding_sam_utils.get_grounding_sam()
    )
    print("getting segmentation mask...")
    input_image_mask = grounding_sam_utils.get_seg_mask(
        image=input_image,
        label=[f"a {input_image_label}."],
        threshold=detection_threshold,
        detector_model=detector_model,
        segmentator_model=segmentator_model,
        segment_processor=segment_processor,
    )
    input_image_mask_arr = general_utils.image_to_tensor(input_image_mask).cuda()

    del (
        labeler_model,
        labeler_processor,
        detector_model,
        segmentator_model,
        segment_processor,
    )
    torch.cuda.empty_cache()

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
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
    )

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    unet.to(accelerator.device, dtype=weight_dtype)

    if args.pretrained_vae_model_name_or_path is None:
        vae.to(accelerator.device, dtype=torch.float32)
    else:
        vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # now we will add new LoRA weights to the attention layers
    # Set correct lora layers
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet.add_adapter(unet_lora_config)

    # The text encoder comes from 🤗 transformers, we will also attach adapters to it.
    if args.train_text_encoder:
        # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one = get_peft_model(text_encoder_one, text_lora_config)
        text_encoder_two = get_peft_model(text_encoder_two, text_lora_config)
        text_encoder_one.add_adapter("a", text_lora_config)
        text_encoder_two.add_adapter("b", text_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder attn layers
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None

            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model)
                        )
                elif isinstance(
                    unwrap_model(model), type(unwrap_model(text_encoder_one))
                ):
                    text_encoder_one_lora_layers_to_save = (
                        convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model)
                        )
                    )
                elif isinstance(
                    unwrap_model(model), type(unwrap_model(text_encoder_two))
                ):
                    text_encoder_two_lora_layers_to_save = (
                        convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model)
                        )
                    )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_two))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, _ = LoraLoaderMixin.lora_state_dict(input_dir)
        unet_state_dict = {
            f'{k.replace("unet.", "")}': v
            for k, v in lora_state_dict.items()
            if k.startswith("unet.")
        }
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(
            unet_, unet_state_dict, adapter_name="default"
        )
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if args.train_text_encoder:
            _set_state_dict_into_text_encoder(
                lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_
            )

            _set_state_dict_into_text_encoder(
                lora_state_dict,
                prefix="text_encoder_2.",
                text_encoder=text_encoder_two_,
            )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [unet_]
            if args.train_text_encoder:
                models.extend([text_encoder_one_, text_encoder_two_])
            cast_training_params(models, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [unet]
        if args.train_text_encoder:
            models.extend([text_encoder_one, text_encoder_two])
        cast_training_params(models, dtype=torch.float32)

    # Optimizer creation
    optimizer_class = torch.optim.AdamW
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if args.train_text_encoder:
        params_to_optimize = (
            params_to_optimize
            + list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
            + list(filter(lambda p: p.requires_grad, text_encoder_two.parameters()))
        )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.num_train_epochs * args.gradient_accumulation_steps,
        # num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        (
            unet,
            text_encoder_one,
            text_encoder_two,
            optimizer,
            lr_scheduler,
        ) = accelerator.prepare(
            unet,
            text_encoder_one,
            text_encoder_two,
            optimizer,
            lr_scheduler,
        )
    else:
        unet, optimizer, lr_scheduler = accelerator.prepare(
            unet, optimizer, lr_scheduler
        )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    if args.resume_from_checkpoint:
        accelerator.print(f"Resuming from checkpoint {args.resume_from_checkpoint}")
        accelerator.load_state(f"{args.resume_from_checkpoint}")

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.num_train_epochs),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    assert (
        args.criterion_type != CriterionType.ir_dino_ensemble.value
        and args.ir_features_weight == 1.0
        and args.dino_features_weight == 1.0
    ) or (args.criterion_type == CriterionType.ir_dino_ensemble.value)

    # get criterion
    if (
        args.criterion_type == CriterionType.ir_features.value
        or args.criterion_type == CriterionType.ir_dino_ensemble.value
    ):
        ir_criterion = get_criterion(CriterionType.ir_features.value)
    if (
        args.criterion_type == CriterionType.classifier_features.value
        or args.criterion_type == CriterionType.ir_dino_ensemble.value
    ):
        dino_criterion = get_criterion(CriterionType.classifier_features.value)
    if args.criterion_type == CriterionType.alpha_clip.value:
        alpha_clip_criterion = get_criterion(CriterionType.alpha_clip.value)

    # get model
    if (
        args.criterion_type == CriterionType.ir_features.value
        or args.criterion_type == CriterionType.ir_dino_ensemble.value
    ):
        ir_feature_extractor, feature_extractor_transforms = (
            ir_features_utils.get_ir_model_and_transforms(
                "IR_dependencies/ir_features.pth",
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
            dino_utils.get_model_and_transforms_configs(
                args.classifier_model_name
            )
        )

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
        dino_subject_image_input = dino_utils.prepare_for_dino(
            subject_image_arr, transforms_configs
        ).cuda()
        with torch.no_grad():
            dino_subject_image_features = dino_utils.get_dino_features(
                classifier, dino_subject_image_input
            )

    pipeline_inversion, pipeline_inference = general_utils.get_pipelines(
        args.pretrained_model_name_or_path,
        unwrap_model(text_encoder_one),
        unwrap_model(text_encoder_two),
        vae,
        unwrap_model(unet),
        accelerator.device,
    )

    inversion_prompt = f"a photo of a {input_image_label}"
    generation_prompt = f"a photo of a {subject_image_label}"

    inversion_config, last_latent = get_inversion_config_and_last_latent(
        args, input_image, pipeline_inversion, pipeline_inference, inversion_prompt
    )

    bg_losses = []
    ir_losses = []
    dino_losses = []
    total_losses = []
    best_loss = float("inf")
    best_image = None
    early_stopping_count = 0
    global_step = 0
    for epoch in range(args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            text_encoder_two.train()
        train_loss = 0.0
        generator = torch.Generator(device=accelerator.device)
        generator.manual_seed(args.seed)
        with accelerator.accumulate(unet):
            image = generate_image(
                args.inversion_type,
                pipeline_inference,
                generation_prompt,
                last_latent,
                inversion_config,
                generator,
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

            if args.bg_mse_loss_weight > 0.0:
                bg_loss = general_utils.get_bg_mse_loss(
                    image_out, input_image_arr, input_image_mask_arr
                )
                bg_losses.append(bg_loss.detach().item())
                loss += args.bg_mse_loss_weight * bg_loss

            total_losses += [loss.detach().item()]

            best_loss, early_stopping_count = update_early_stopping_count(
                args.early_stopping_threshold_percentage,
                early_stopping_count,
                best_loss,
                loss,
            )

            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)

            if args.log_every_epoch and accelerator.is_main_process:
                image_out = general_utils.numpy_to_pil(
                    image_out.detach().permute(0, 2, 3, 1).cpu().numpy()
                )[0]
                image_out.save(f"{args.output_dir}/epoch_{epoch}.png")

                general_utils.log_all_losses(
                    {
                        "total_losses": total_losses,
                        **(
                            {"bg_losses": bg_losses}
                            if args.bg_mse_loss_weight > 0.0
                            else {}
                        ),
                        **(
                            {"ir_losses": ir_losses, "dino_losses": dino_losses}
                            if args.criterion_type
                            == CriterionType.ir_dino_ensemble.value
                            else {}
                        ),
                    },
                    args.output_dir,
                )

            if loss <= best_loss:
                if type(image_out) != Image.Image:
                    image_out = general_utils.numpy_to_pil(
                        image_out.detach().permute(0, 2, 3, 1).cpu().numpy()
                    )[0]
                best_image = image_out

            if early_stopping_count >= args.early_stopping_max_count:
                print(f"Early stopping at epoch {epoch}")
                break

    if args.save_loss_threshold is None or best_loss < args.save_loss_threshold:
        # save losses array to disk as an npy file
        np.save(f"{args.output_dir}/total_losses.npy", np.array(total_losses))
        if args.criterion_type == CriterionType.ir_dino_ensemble.value:
            np.save(f"{args.output_dir}/ir_losses.npy", np.array(ir_losses))
            np.save(f"{args.output_dir}/dino_losses.npy", np.array(dino_losses))
        if args.bg_mse_loss_weight > 0.0:
            np.save(f"{args.output_dir}/bg_losses.npy", np.array(bg_losses))

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
    else:
        print("Loss threshold not met. Not saving results.")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
