# Single Image Iterative Subject-driven Generation and Editing

> **Yair Shpitzer, Gal Chechik, Idan Schwartz**
>
> Personalizing image generation and editing is particularly challenging when we only have a few images of the subject, or even a single image. A common approach to personalization is concept learning, which can integrate the subject into existing models relatively quickly, but produces images whose quality tends to deteriorate quickly when the number of subject images is small. Quality can be improved by pre-training an encoder, but training restricts generation to the training distribution, and is time consuming. It is still an open hard challenge to personalize image generation and editing from a single image without training. Here, we present SISO, a novel, training-free approach based on optimizing a similarity score with an input subject image. More specifically, SISO iteratively generates images and optimizes the model based on loss of similarity with the given subject image until a satisfactory level of similarity is achieved, allowing plug-and-play optimization to any image generator. We evaluated SISO in two tasks, image editing and image generation, using a diverse data set of personal subjects, and demonstrate significant improvements over existing methods in image quality, subject fidelity, and background preservation. 

<a href=""><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 

<a href=""><img src="" height=20.5></a>

<p align="center">
<img src="figures/teaser.png" width="800px"/>
</p>

## Description

Official implementation of *Single Image Iterative Subject-driven Generation and Editing*. 

## Setup
Clone the repository and navigate into the directory:
```
git clone https://github.com/yairshp/SISO.git
cd SISO
```

Create a conda environment using the following command:
```
conda env create -f environment.yml
```

Install the dependencies for the IR feature extractor:

```
cd IR_dependencies/open_clip_280
pip install -e .
cd ../open_clip_280_overlap
pip install -e .
```

Now activate your environment:
```
conda activate siso
```

Finally, download IR model weights from [here](https://www.kaggle.com/datasets/louieshao/guieweights0732?resource=download).

## Run SISO for Generation

### FLUX
```
python siso_generation_flux.py --subject_image_path example_images/dog_subject.png --prompt a\ photo\ of\ a\ dog --train_text_encoder --output_dir logs/dog_generation --lr_warmup_steps 0 --lr_scheduler constant --train_batch_size 1 --resolution 512 --pretrained_model_name_or_path black-forest-labs/FLUX.1-schnell  --num_train_epochs 50 --early_stopping_threshold_percentage 3 --early_stopping_max_count 7 --num_inference_steps 1 --learning_rate 2e-4 --seed=42 --save_weights --weights_output_dir weights/dog --ir_features_path <path_to_IR_weights> --mixed_precision bf16
```

### SDXL

```
python siso_generation_flux.py --subject_image_path example_images/dog_subject.png --prompt a\ photo\ of\ a\ dog --train_text_encoder --output_dir logs/dog_generation --lr_warmup_steps 0 --lr_scheduler constant --train_batch_size 1 --resolution 512 --pretrained_model_name_or_path stabilityai/sdxl-turbo --num_train_epochs 50 --early_stopping_threshold_percentage 3 --early_stopping_max_count 7 --num_inference_steps 1 --learning_rate 2e-4 --seed=42 --save_weights --weights_output_dir weights/dog --ir_features_path <path_to_IR_weights>
```

### Sana

```
python siso_generation_sana.py --subject_image_path example_images/dog_subject.png --prompt a\ photo\ of\ a\ dog --output_dir logs/dog_generation --lr_warmup_steps 0 --lr_scheduler constant --train_batch_size 1 --resolution 512 --pretrained_model_name_or_path Efficient-Large-Model/Sana_1600M_512px_diffusers  --num_train_epochs 50 --early_stopping_threshold_percentage 3 --early_stopping_max_count 5 --num_inference_steps 20 --learning_rate 8e-4 --seed=42 --save_weights --weights_output_dir weights/dog --ir_features_path <path_to_IR_weights>  --num_grad_steps 3 
```


To use the LoRA weights for inference on another prompt, follow the instructions [here](https://huggingface.co/docs/diffusers/en/using-diffusers/loading_adapters#lora).


## Run SISO for Editing

### SDXL

```
 python siso_editing_sdxl.py --output_dir logs/dog_editing --seed=42 --lr_warmup_steps 0 --lr_scheduler constant --learning_rate 3e-4 --train_batch_size 1 --resolution 512 --pretrained_model_name_or_path stabilityai/sdxl-turbo --num_train_epochs 50 --bg_mse_loss_weight 10. --ir_features_weight 1.0 --dino_features_weight 1.0 --early_stopping_threshold_percentage 3 --early_stopping_max_count 7 --input_image_path example_images/dog_input.png --subject_image_path example_images/dog_subject.png --ir_features_path <path_to_IR_weights>
```