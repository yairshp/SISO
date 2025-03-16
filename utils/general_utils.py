import torch
import requests
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

import utils.renoise_inversion_utils as renoise_inversion_utils


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def load_image(image_path_or_url: str):
    """
    Load an image from a file path or a URL.
    """
    if image_path_or_url.startswith("http"):

        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path_or_url)
    image = image.convert("RGB")
    return image


def image_to_tensor(image):
    image = TF.pil_to_tensor(image).float().unsqueeze(0) / 255.0
    return image


def resize_and_center_crop(img, target_size=(512, 512)):
    # Calculate the aspect ratio and resize the image
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]

    if img_ratio > target_ratio:
        # Image is wider than target aspect ratio
        new_height = target_size[1]
        new_width = int(new_height * img_ratio)
    else:
        # Image is taller than target aspect ratio
        new_width = target_size[0]
        new_height = int(new_width / img_ratio)

    img_resized = img.resize((new_width, new_height))

    # Calculate the cropping box
    left = (new_width - target_size[0]) / 2
    top = (new_height - target_size[1]) / 2
    right = (new_width + target_size[0]) / 2
    bottom = (new_height + target_size[1]) / 2

    # Crop the image
    img_cropped = img_resized.crop((left, top, right, bottom))

    return img_cropped


def get_bg_mse_loss(generated_image, input_image, seg_mask):
    # Expand the mask to match the dimensions of the images
    expanded_mask = seg_mask.repeat(1, 3, 1, 1)

    generated_image_outside_mask = generated_image * (1 - expanded_mask)
    bg_image_outside_mask = input_image * (1 - expanded_mask)

    squared_diff = (generated_image_outside_mask - bg_image_outside_mask) ** 2
    sum_squared_diff = squared_diff.sum()
    num_unmasked_pixels = torch.sum(1 - expanded_mask)
    mse = sum_squared_diff / num_unmasked_pixels

    return mse


def get_pipelines(
    model_name_or_path,
    text_encoder_one,
    text_encoder_two,
    vae,
    unet,
    device,
):
    pipeline_inversion, pipeline_inference = (
        renoise_inversion_utils.get_renoise_inversion_pipes(
            model_name=model_name_or_path,
            vae=vae,
            text_encoder_one=text_encoder_one,
            text_encoder_two=text_encoder_two,
            unet=unet,
            device=device,
        )
    )

    return pipeline_inversion, pipeline_inference


def apply_seg_mask_on_generated_image(generated_image, mask):
    # both generated_image and mask are tensors
    mask = mask.repeat(1, 3, 1, 1)
    return generated_image * mask


def plot_losses(losses, path):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Losses")
    plt.savefig(path)


def log_all_losses(losses_dict, output_dir):
    for loss_name, losses in losses_dict.items():
        plot_losses(losses, f"{output_dir}/{loss_name}.png")
