import torch
import kornia.geometry.transform as KTF
from transformers import AutoProcessor, GroundingDinoForObjectDetection
from torchvision.transforms.functional import to_pil_image


def get_grounding_dino_model_and_processor():
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    model = GroundingDinoForObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-tiny"
    )

    return model, processor


def get_bbox(model, processor, image, object_to_detect):
    image_input = to_pil_image(image.squeeze())
    inputs = processor(
        images=image_input, text=[f"a {object_to_detect}."], return_tensors="pt"
    )
    outputs = model(**inputs)
    target_sizes = torch.tensor([image_input.size[::-1]])
    results = processor.image_processor.post_process_object_detection(
        outputs, threshold=0.35, target_sizes=target_sizes
    )[0]
    best_score = 0
    best_box = None
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = [round(i, 1) for i in box.tolist()]
        if score > best_score:
            best_score = score
            best_box = box
    best_box = [
        (best_box[0], best_box[1]),
        (best_box[2], best_box[1]),
        (best_box[2], best_box[3]),
        (best_box[0], best_box[3]),
    ]
    return torch.tensor(best_box).unsqueeze(0).to("cuda")


def crop_by_bbox(image, bbox):
    cropped = KTF.crop_by_indices(image, bbox)
    return cropped
