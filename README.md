# Single Image Iterative Subject-driven Generation and Editing

This is an implementation of the paper *Single Image Iterative Subject-driven Generation and Editing*. 

## Setup

First, create a conda environment using the following command:
```
conda env create -f environment.yml
```

Second, install the dependencies for the IR feature extractor:

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

Finally, download IR model from [here](https://www.kaggle.com/datasets/louieshao/guieweights0732?resource=download). Rename the downloaded file to **ir_features.pth** and put it under the folder **IR_dependencies**

## Run SISO for Generation
```
python siso_generation.py --subject_image_path_or_url example_images/dog_subject.png --prompt a\ photo\ of\ a\ dog --train_text_encoder --output_dir logs/dog_generation --lr_warmup_steps 0 --lr_scheduler constant --train_batch_size 1 --resolution 512 --pretrained_model_name_or_path stabilityai/sdxl-turbo --num_train_epochs 50 --criterion_type ir_dino_ensemble --classifier_model_name vit_large_patch14_dinov2.lvd142m --early_stopping_threshold_percentage 3 --early_stopping_max_count 7 --num_inference_steps 4 --learning_rate 2e-4 --seed=42 --save_weights --weights_output_dir weights/dog
```


To use the LoRA weights for inference on another prompt, follow the instructions [here](https://huggingface.co/docs/diffusers/en/using-diffusers/loading_adapters#lora).


## Run SISO for Editing
```
python siso_editing.py --output_dir logs/dog_editing --seed=42 --lr_warmup_steps 0 --lr_scheduler constant --learning_rate 3e-4 --train_batch_size 1 --resolution 512 --pretrained_model_name_or_path stabilityai/sdxl-turbo --num_train_epochs 50 --criterion_type ir_dino_ensemble --classifier_model_name vit_large_patch14_dinov2.lvd142m --inversion_type renoise --bg_mse_loss_weight 10. --ir_features_weight 1.0 --dino_features_weight 1.0 --early_stopping_threshold_percentage 3 --early_stopping_max_count 7 --input_image_path_or_url example_images/dog_input.png --subject_image_path_or_url example_images/dog_subject.png
```