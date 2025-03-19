import torch
from PIL import Image
from argparse import ArgumentParser
from diffusers import StableDiffusionXLPipeline, FluxPipeline, SanaPipeline


def get_pipeline(model_name: str, device: torch.device):
    if "sdxl" in model_name:
        pipeline = StableDiffusionXLPipeline.from_pretrained(model_name).to(device)
    elif "FLUX" in model_name:
        pipeline = FluxPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16,).to(device)
    elif "Sana" in model_name:
        pipeline = SanaPipeline.from_pretrained(model_name, variant="fp16", torch_dtype=torch.float16,).to(device)
        pipeline.vae.to(torch.bfloat16)
        pipeline.text_encoder.to(torch.bfloat16)
    else:
        raise ValueError("Model not supported")
    return pipeline

    
def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--lora_weights_path", type=str, required=True)
    parser.add_argument("--num_inference_steps", type=int, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--output_path", type=str, default="output.png")
    parser.add_argument("--device_id", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

    
def main():
    args = get_args()
    device = torch.device(f"cuda:{args.device_id}")
    pipeline = get_pipeline(args.model_name, device)
    pipeline.load_lora_weights(args.lora_weights_path)

    generator = torch.Generator(device)
    generator.manual_seed(args.seed)

    image = pipeline(
        args.prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        width=512,
        height=512,
    ).images[0]

    image.save(args.output_path)


if __name__ == "__main__":
    main()