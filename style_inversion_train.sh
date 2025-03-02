gpu_ids=$1
image_name=$2
domain_class_token=$3
domain_embed_scale=$4
batch_size=$5


# YAML配置内容
config="compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: '$gpu_ids'
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1 
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false"

# 将配置内容写入到文件中
echo "$config" > ./accelerate_config/config_$image_name.yaml

echo "配置已写入到 config_$image_name.yaml 文件中，gpu_ids 参数为: $gpu_ids"

CUDA_VISIBLE_DEVICES=1 accelerate  launch --config_file ./accelerate_config/config_$image_name.yaml  --main_process_port 29531 pretrain_inversion_appearance.py \
  --checkpointing_steps=100 \
  --clip_model_name_or_path="ViT-B-32::Weights/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin" \
  --dataset_class="XXX" \
  --domain_class_token=$domain_class_token \
  --shape_class_token="shape" \
  --appearance_class_token="appearance" \
  --domain_embed_scale=$domain_embed_scale \
  --exp_name="Appearance_Inversion" \
  --exp_desc="Teacher-Student-Architecture-V2" \
  --iterable_dataset \
  --learning_rate=1e-5 --scale_lr \
  --log_steps=20 \
  --max_train_steps=1000 \
  --mixed_precision="fp16" \
  --output_dir="Outs/$image_name-$domain_embed_scale-$batch_size" \
  --placeholder_token="*s" \
  --placeholder_token_shape="*m" \
  --placeholder_token_appearance="*a" \
  --pretrained_model_name_or_path="Weights/CompVis/stable-diffusion-v1-4" \
  --prompt_template="joint" \
  --reg_lambda=0.01 \
  --resolution=512 \
  --student_prompt="a $domain_class_token in the style of *a" \
  --train_batch_size=$batch_size \
  --origin_img_path="data/style/$image_name" \
  --gradient_accumulation_steps=2 \
  --loss_diff_student_weight=1.0 \
  --loss_shape_weight=0.0 \
  --loss_appearance_weight=0.0 \
  --temp_dino_or_e4t="e4t" \
  --unfreeze_clip_vision \
  --is_need_augment \
  --use_8bit_adam