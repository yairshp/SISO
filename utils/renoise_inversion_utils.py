import torch
from diffusers.utils.torch_utils import randn_tensor

from third_party.renoise_inversion.src.schedulers.ddim_scheduler import MyDDIMScheduler
from third_party.renoise_inversion.src.schedulers.euler_scheduler import (
    MyEulerAncestralDiscreteScheduler,
)
from third_party.renoise_inversion.src.pipes.sdxl_inversion_pipeline import (
    SDXLDDIMPipeline,
)
from third_party.renoise_inversion.src.pipes.sd_inversion_pipeline import SDDDIMPipeline
from third_party.renoise_inversion.src.config import RunConfig
from third_party.renoise_inversion.src.eunms import Model_Type, Scheduler_Type
from third_party.renoise_inversion.src.utils.enums_utils import (
    model_type_to_size,
)
from local_pipelines.pipeline_stable_diffusion_xl_img2img_with_grads import (
    StableDiffusionXLImg2ImgPipelineWithGrads,
)


def create_noise_list(model_type, length, generator):
    img_size = model_type_to_size(model_type)
    VQAE_SCALE = 8
    latents_size = (1, 4, img_size[0] // VQAE_SCALE, img_size[1] // VQAE_SCALE)
    return [
        randn_tensor(
            latents_size,
            dtype=torch.float32,
            device=torch.device("cuda:0"),
            generator=generator,
        )
        for i in range(length)
    ]


def get_renoise_inversion_pipes(
    vae,
    text_encoder_one,
    text_encoder_two,
    unet,
    model_name="stabilityai/sdxl-turbo",
    device="cuda",
):
    pipe_inference = StableDiffusionXLImg2ImgPipelineWithGrads.from_pretrained(
        model_name,
        vae=vae,
        text_encoder_one=text_encoder_one,
        text_encoder_two=text_encoder_two,
        unet=unet,
        use_safetensors=True,
        safety_checker=None,
    ).to(device)
    pipe_inversion = SDXLDDIMPipeline(**pipe_inference.components)

    pipe_inference.scheduler = MyEulerAncestralDiscreteScheduler.from_config(
        pipe_inference.scheduler.config
    )
    pipe_inversion.scheduler = MyEulerAncestralDiscreteScheduler.from_config(
        pipe_inversion.scheduler.config
    )
    return pipe_inversion, pipe_inference


def get_inversion_config(inversion_max_step=0.75, model_name="stabilityai/sdxl-turbo"):
    if "sdxl" in model_name:
        model_type = Model_Type.SDXL_Turbo
        scheduler_type = Scheduler_Type.EULER
        perform_noise_correction = True
        noise_regularization_lambda_ac = 20.0
    else:
        model_type = Model_Type.SD21_Turbo
        scheduler_type = Scheduler_Type.DDIM
        perform_noise_correction = False
        noise_regularization_lambda_ac = 0.0
    config = RunConfig(
        model_type=model_type,
        scheduler_type=scheduler_type,
        inversion_max_step=inversion_max_step,
        perform_noise_correction=perform_noise_correction,
        noise_regularization_lambda_ac=noise_regularization_lambda_ac,
    )
    return config


def invert(pipe_inversion, pipe_inference, init_image, prompt, cfg, is_sdxl=True):
    generator = torch.Generator().manual_seed(cfg.seed)
    if is_sdxl:
        noise = create_noise_list(
            cfg.model_type, cfg.num_inversion_steps, generator=generator
        )
        pipe_inversion.scheduler.set_noise_list(noise)
        pipe_inference.scheduler.set_noise_list(noise)

    pipe_inversion.cfg = cfg
    pipe_inference.cfg = cfg
    res = pipe_inversion(
        prompt=prompt,
        num_inversion_steps=cfg.num_inversion_steps,
        num_inference_steps=cfg.num_inference_steps,
        generator=generator,
        image=init_image,
        guidance_scale=cfg.guidance_scale,
        strength=cfg.inversion_max_step,
        denoising_start=1.0 - cfg.inversion_max_step,
        num_renoise_steps=cfg.num_renoise_steps,
    )
    latents = res[0][0]
    return latents
