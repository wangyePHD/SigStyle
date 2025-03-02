#!/bin/bash

name="tower"

pretrained_folder="XXX"
result_folder="results/$name/"

edit_prompt="a photo of the Eiffel Tower, in the style of *a"
reference_img="data/style/05.jpg"
edit_img="data/content/tower.png"
source_prompt="a photo of the Eiffel Tower"


CUDA_VISIBLE_DEVICES=1 python inference_student_appearance.py \
  --pretrained_model_name_or_path "$pretrained_folder" \
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

