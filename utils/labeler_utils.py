import requests
import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor


def get_labeler_model_and_processor(device: str = "cuda"):
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf"
    ).to(device)
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    return model, processor


def get_label(image, model, processor):
    query = "USER: <image>\nwhat object is in the image? describe in no more than two words. ASSISTANT:"
    inputs = processor(text=query, images=image, return_tensors="pt").to("cuda")
    generate_ids = model.generate(**inputs, max_new_tokens=50)
    answer = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    answer = answer.split(":")[-1].strip().lower()
    return answer
