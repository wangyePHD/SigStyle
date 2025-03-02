#!/bin/bash

name="tower"

base_folder="/data1/ye_project/OSTAF_Output/SD1.4/1/appearance-inversion-05-0.1-1_2025-03-02-11.59/weight/"
result_folder="/data1/ye_project/SigStyle/results/$name/"

# edit_prompt="a photo of the taj mahal, in the style of *a"
edit_prompt="a photo of the Eiffel Tower, in the style of *a"
# edit_prompt="a photo of a horse, in the style of *a" 
reference_img="/data1/ye_project/SigStyle/data/style/05.jpg"
edit_img="/data1/ye_project/SigStyle/data/content/tower.png"
source_prompt="a photo of the Eiffel Tower"

# source_prompt="a building surrounded by trees and rivers"
# source="a photo of a sunset over the ocean with waves in the foreground"
# source_prompt="a girl"
# source_prompt="a photo of the Times Square"
# source_prompt="a photo of river and mountain"
# source_prompt="a photo of the Great Wall"
# source_prompt="a photo of the taj mahal"
# source_prompt="a photo of a horse"
# source_prompt="a bridge"

CUDA_VISIBLE_DEVICES=1 python inference_student_appearance.py \
  --pretrained_model_name_or_path "$base_folder" \
  --teacher_model_name_or_path "None" \
  --prompt "$edit_prompt"\
  --num_images_per_prompt 1 \
  --scheduler_type "ddim" \
  --image_path_or_url "$reference_img" \
  --num_inference_steps 50 \
  --guidance_scale 7.5 \
  --target_folder "$result_folder" \
  --seed 42 \
  --edit_image_path_or_url "$edit_img" \
  --source_prompt "$source_prompt" \
  --edit_mask "$edit_mask" \
  --time 15 \
  --img_pre_name $name \

