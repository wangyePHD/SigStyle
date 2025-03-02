import argparse
import os
import random
import sys

from packaging import version
from tqdm.auto import tqdm
import itertools
import json

from PIL import Image
from einops import rearrange
from torch.nn import functional as F
from torch.utils.data import DataLoader

import numpy as np
import torch.nn as nn
import torch
import wandb
import math

from accelerate import Accelerator
from accelerate.utils import set_seed

from diffusers import DDPMScheduler, AutoencoderKL, DDIMScheduler, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available

from dataset import ShapeInvDataset, AppearanceInvDataset
from shapeinv.encoder import ShapeInvEncoder
from shapeinv.models.modeling_clip import CLIPTextModel
from shapeinv.pipeline_stable_diffusion_e4t import *
from shapeinv.utils import *
from transformers import CLIPTokenizer

from utils_wy import *
from loguru import logger
from filter import MLP


def visualize_when_training(original_samples, latent_noisy,timesteps, pre_noise, noise_scheduler):
    
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(pre_noise)
    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
    
    pred_x0 = (latent_noisy - sqrt_one_minus_alpha_prod * pre_noise) / sqrt_alpha_prod
    
    return pred_x0
    
def decode_latents(latents, vae):

    latents = 1 / vae.config.scaling_factor * latents
    latents = latents.to(vae.dtype)
    imgs = vae.decode(latents).sample
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = imgs.permute(0, 2, 3, 1).float()
    return imgs

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    images = images.cpu().numpy()
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def get_logger(path, accelerator):
    if accelerator.is_main_process:
        logger.add(os.path.join(path,'exp_{time}.log'),format="{time} | {level} | {message}",level="INFO")
    return logger

def loguru_info(logger, accelerator, msg):
    if accelerator.is_main_process:
        logger.info(msg)
    else:
        pass


def importance_sampling_fn(t, max_t, alpha):
    """Importance Sampling Function f(t)"""
    return 1 / max_t * (1 + alpha * math.cos(math.pi * t / max_t))


def parse_args():
    
    """
    Parse command line arguments and return the parsed arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # e4t configs
    parser.add_argument("--clip_model_name_or_path", type=str, default="ViT-H-14::laion2b_s32b_b79k", required=False, help="load from open_clip with the format 'arch::version'")
    parser.add_argument("--domain_class_token", type=str, default=None, required=True, help="Coarse-class token such as `face`, `cat`, pr `art`")
    parser.add_argument("--shape_class_token",type=str, default=None, required=True, help="Shape class token")
    parser.add_argument("--appearance_class_token",type=str, default=None, required=True, help="Appearance class token")
    parser.add_argument("--domain_embed_scale", type=float, default=0.1, help="scale of e4t encoder's embedding")
    parser.add_argument("--iterable_dataset", action="store_true", default=False, help="Use iterable dataset in datasets")
    parser.add_argument("--placeholder_token", type=str, default="*s", help="A token to use as a placeholder for the concept.",)
    parser.add_argument("--placeholder_token_shape", type=str, default="*m", help="A token to use as a placeholder for the shape.",)
    parser.add_argument("--placeholder_token_appearance", type=str, default="*a", help="A token to use as a placeholder for the appearance.",)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5", required=False, help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--teacher_e4t_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5", required=False, help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--student_model_init_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5", required=False, help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--prompt_template", type=str, default="joint", help="prompt template type")
    parser.add_argument("--reg_lambda", type=float, default=0.01, help="l2 regularization lambda")
    parser.add_argument("--unfreeze_clip_vision", action="store_true", default=False, help="train clip image encoder as a part of e4t encoder")
    parser.add_argument("--webdataset", action="store_true", default=False, help="load tar files via webdataset")
    # training
    parser.add_argument("--checkpointing_steps", type=int, default=1000, help="Save a checkpoint of the training state every X updates.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--learning_rate", type=float, default=1.6e-5, help="learning rate",)
    parser.add_argument("--log_steps", type=int, default=500, help="sample images ")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_train_steps", type=int, default=30000, help="Total number of training steps to perform. For face, 30,000. For cat, 60,000. For art, 100,000",)
    parser.add_argument("--num_train_epochs", type=int, default=1,)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help=("Whether training should be resumed from a previous checkpoint. Use a path saved by" ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'))
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")
    # log
    parser.add_argument("--n_save_sample", type=int, default=4, help="The number of samples per prompt")
    parser.add_argument("--save_guidance_scale", type=float, default=7.5, help="CFG for save sample.")
    parser.add_argument("--save_inference_steps", type=int, default=50, help="The number of inference steps for save sample.",)
    
    # general
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--output_dir", type=str, default="e4t-model", help="The output directory where the model predictions and checkpoints will be written.", )
    parser.add_argument("--report_to", type=str, default="wandb", choices=["tensorboard", "wandb"])
    parser.add_argument("--revision", type=str, default=None, required=False, help="Revision of pretrained model identifier from huggingface.co/models.", )
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
    # exp
    parser.add_argument("--origin_img_path", type=str, default="/home/wangye/Research2023/ShapeInversion_Data/data/train_origin_img.txt", help="Origin image path.")
    parser.add_argument("--mask_img_path", type=str, default="/home/wangye/Research2023/ShapeInversion_Data/data/train_seg_mask.txt", help="Mask image path.")
    parser.add_argument("--exp_name", type=str, default="e4t", help="Experiment name.")
    parser.add_argument("--exp_desc", type=str, default=None, help="Experiment description.")
    parser.add_argument("--dataset_class", type=str, default="chair")
    parser.add_argument("--loss_diff_student_weight", type=float, default=1.0)
    parser.add_argument("--loss_shape_weight", type=float, default=1.0)
    parser.add_argument("--loss_appearance_weight", type=float, default=1.0)
    parser.add_argument("--loss_mask_weight", type=float, default=0.1)
    parser.add_argument("--training_teacher_only", action="store_true")
    parser.add_argument("--is_need_augment", action="store_true")
    parser.add_argument("--student_prompt", type=str, default="student prompt", help="split with ',' for multiple prompts")
    parser.add_argument("--temp_dino_or_e4t", type=str, default="e4t")
    parser.add_argument("--origin_img_path1",type=str)
    
    args = parser.parse_args()
    
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args



def main():
    
    args = parse_args()
    
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )
    
    post_fix = "_"+obtain_time()
    args.output_dir = args.output_dir+post_fix
    params_folder = args.output_dir
    code_folder = os.path.join(args.output_dir, "code")
    weight_folder = os.path.join(args.output_dir, "weight")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(params_folder, exist_ok=True)
    os.makedirs(code_folder, exist_ok=True)
    os.makedirs(weight_folder, exist_ok=True)
    
    logger = get_logger(args.output_dir, accelerator)
    loguru_info(logger,accelerator,"Start training")
    
    extensions = ('.py', '.sh', '.bash') # 需要保存的文件类型
    source_folder = '.'  # 当前文件夹
    backup_files(source_folder, code_folder, extensions, args)
    backup_params(params_folder, args)
    loguru_info(logger, accelerator, "Backup code and params to folder: {}".format(args.output_dir))
    
    if args.seed is not None:
        set_seed(args.seed)

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    
        
    teacher_unet = load_shapeinv_unet(
        args.pretrained_model_name_or_path,
        ckpt_path=os.path.join(args.teacher_e4t_model_name_or_path, "unet.pt") if os.path.exists(os.path.join(args.teacher_e4t_model_name_or_path, "unet.pt")) else None,
        revision=args.revision
    )
    
    teacher_inv_encoder = load_shapeinv_encoder(
        word_embedding_dim=text_encoder.config.hidden_size,
        block_out_channels=teacher_unet.config.block_out_channels,
        arch=args.clip_model_name_or_path.split("::")[0],
        version=args.clip_model_name_or_path.split("::")[1],
        freeze_clip_vision=not args.unfreeze_clip_vision,
        ckpt_path=os.path.join(args.teacher_e4t_model_name_or_path, "encoder.pt") if os.path.exists(os.path.join(args.teacher_e4t_model_name_or_path, "encoder.pt")) else None,
    )
    
    student_unet = load_shapeinv_unet(
        args.pretrained_model_name_or_path,
        ckpt_path=os.path.join(args.pretrained_model_name_or_path, "weight_offsets.pt") if os.path.exists(os.path.join(args.pretrained_model_name_or_path, "weight_offsets.pt")) else None,
        revision=args.revision
    )
    
    
    loguru_info(logger, accelerator, "teacher 和 student模型加载成功！！！")
    
    
    appearance_filter = MLP(input_size=768, hidden_size=768, output_size=768) 
    
    loguru_info(logger, accelerator, "Added placeholder token to tokenizer")
    
    # * Add the placeholder tokens in tokenizer
    num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    num_added_tokens_shape = tokenizer.add_tokens(args.placeholder_token_shape)
    num_added_tokens_appearance = tokenizer.add_tokens(args.placeholder_token_appearance)
    
    if num_added_tokens == 0 or num_added_tokens_shape == 0 or num_added_tokens_appearance == 0:
        raise ValueError(f"The tokenizer already contains the token {args.placeholder_token} or {args.placeholder_token_shape} or {args.placeholder_token_appearance}. Please pass a different `placeholder_token` that is not already in the tokenizer.")
    
    # * convert tokens to ids
    placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)
    placeholder_token_shape_id = tokenizer.convert_tokens_to_ids(args.placeholder_token_shape)
    placeholder_token_appearance_id = tokenizer.convert_tokens_to_ids(args.placeholder_token_appearance)
    
    loguru_info(logger, accelerator, f"placeholder_token:{args.placeholder_token}. placeholder_token_id: {placeholder_token_id}")
    loguru_info(logger, accelerator, f"placeholder_token_shape:{args.placeholder_token_shape}. placeholder_token_shape_id: {placeholder_token_shape_id}")
    loguru_info(logger, accelerator, f"placeholder_token_appearance:{args.placeholder_token_appearance}. placeholder_token_appearance_id: {placeholder_token_appearance_id}")
    
    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))
    
    text_encoder.requires_grad_(False)
    teacher_unet.requires_grad_(False)
    teacher_inv_encoder.requires_grad_(False)
    
    # default x_former is unable to use
    if is_xformers_available() and args.enable_xformers_memory_efficient_attention:
        import xformers
        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            print("[WARNING] xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details.")
        student_unet.enable_xformers_memory_efficient_attention()
        loguru_info(logger,accelerator,"Using xFormers!")
    else:
        loguru_info(logger, accelerator, "xFormers is not available. Memory efficient attention layers will not be used.")
    
    # * initialize the optimizer, add encoder params and weight offsets of student models
    optim_params = []
    for n, p in student_unet.named_parameters():
        if "up_blocks.1" in n or "up_blocks.2" in n or "up_blocks.3" in n:
            if p.requires_grad:
                optim_params += [p]
                print(n)
    
    total_unet_params = sum(p.numel() for p in optim_params)
    loguru_info(logger, accelerator, f"Total number of UNet parameters: {total_unet_params * 1.e-6:.2f} M")    
    # * add appearance filter params
    optim_params += [p for p in appearance_filter.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in optim_params)
    loguru_info(logger, accelerator, f"Total number of ALL trainable parameters: {total_params * 1.e-6:.2f} M")

    #* define dataset
    if os.path.exists(args.origin_img_path+".jpg"):
        full_path = args.origin_img_path+".jpg"
        # if args.origin_img_path1:
        #     full_path1 = args.origin_img_path1+".jpg"
        # write the imgpath into txt *100 times
        with open(full_path.replace(".jpg", ".txt"), "w") as f:
            for i in range(100):
                f.write(full_path+"\n")    
                # f.write(full_path1+"\n")    
        
        origin_img_path = full_path.replace(".jpg", ".txt")    

    if os.path.exists(args.origin_img_path+".png"):
        full_path = args.origin_img_path+".png"
        # write the imgpath into txt *100 times
        # if args.origin_img_path1:
        #     full_path1 = args.origin_img_path1+".png"
        with open(full_path.replace(".png", ".txt"), "w") as f:
            for i in range(100):
                f.write(full_path+"\n")    
                # f.write(full_path1+"\n")    
        origin_img_path = full_path.replace(".png", ".txt")    
    
    if os.path.exists(args.origin_img_path+".jpeg"):
        full_path = args.origin_img_path+".jpeg"
        # if args.origin_img_path1:
        #     full_path1 = args.origin_img_path1+".jpeg"
        # write the imgpath into txt *100 times
        with open(full_path.replace(".jpeg", ".txt"), "w") as f:
            for i in range(100):
                f.write(full_path+"\n")    
                # f.write(full_path1+"\n")    
        origin_img_path = full_path.replace(".jpeg", ".txt")    
    
    
    
    train_dataset = ShapeInvDataset(origin_img_path, resolution=512, is_need_aug=args.is_need_augment)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    loguru_info(logger, accelerator, f"Train dataset length: {len(train_dataset)}")
    loguru_info(logger, accelerator, f"Train dataloader length: {len(train_dataloader)}")
    
    
    # log the dataset length
    loguru_info(logger, accelerator, f"Dataset length: {len(train_dataset)}")
    
    if args.scale_lr:
        learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                learning_rate, args.gradient_accumulation_steps, accelerator.num_processes, args.train_batch_size, args.learning_rate))
        args.learning_rate = learning_rate
    
    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(teacher_unet).dtype != torch.float32:
            raise ValueError(
                f"Unet loaded as datatype {accelerator.unwrap_model(teacher_unet).dtype}. {low_precision_error_string}"
            )

    if accelerator.unwrap_model(teacher_inv_encoder).dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {accelerator.unwrap_model(teacher_inv_encoder).dtype}."
            f" {low_precision_error_string}"
        )

    if accelerator.unwrap_model(student_unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(student_unet).dtype}. {low_precision_error_string}"
        )
        
    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    
    # define optimizer
    optimizer = optimizer_class(
        optim_params,
        lr=args.learning_rate,
    )
    
    # define scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    
    teacher_unet, teacher_inv_encoder, student_unet, appearance_filter, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
    teacher_unet, teacher_inv_encoder, student_unet, appearance_filter, optimizer, lr_scheduler, train_dataloader
    )
    
    # For mixed precision training we cast the unet and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    # Move vae and unet to device and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    
    
 
    args.num_train_epochs = 250000000000000000000

    
    # ! Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    loguru_info(logger, accelerator, "***** Running training *****")
    if not args.iterable_dataset and not args.webdataset:
        loguru_info(logger, accelerator, f"  Num examples = {len(train_dataset)}")
        loguru_info(logger, accelerator, f"  Num Epochs = {args.num_train_epochs}")
    
    loguru_info(logger, accelerator, f"  Instantaneous batch size per device = {args.train_batch_size}")
    loguru_info(logger, accelerator, f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    loguru_info(logger, accelerator, f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    loguru_info(logger, accelerator, f"  Total optimization steps = {args.max_train_steps}")
    
   
    # define sample for generation images
    @torch.no_grad()
    def sample(images, masks, step):
        images_to_log = []
        # to pil
        x_samples = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        for x_sample in x_samples:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(x_sample.astype(np.uint8))
            images_to_log.append(img)

        masks_to_log = []
        masks = torch.clamp((masks + 1.0) / 2.0, min=0.0, max=1.0)
        for mask in masks:
            mask = 255. * rearrange(mask.cpu().numpy(), 'c h w -> h w c')
            mask = Image.fromarray(mask.astype(np.uint8))
            masks_to_log.append(mask)
        
        
        random_indices = random.sample(range(len(masks_to_log)), min(len(images_to_log), args.n_save_sample))
        selected_masks_to_log = [masks_to_log[i] for i in random_indices]
        selected_images_to_log = [images_to_log[i] for i in random_indices]
        
        input_grid = image_grid(selected_images_to_log, rows=1, cols=len(selected_images_to_log))
        input_mask = image_grid(selected_masks_to_log, rows=1, cols=len(selected_images_to_log))
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()    
    

    def save_weights(step):
        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            unet_model = accelerator.unwrap_model(student_unet, keep_fp32_wrapper=True)
            appearance_filter_model = accelerator.unwrap_model(appearance_filter, keep_fp32_wrapper=True)

            save_dir = os.path.join(weight_folder, f"{step}")
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                json.dump(args.__dict__, f, indent=2)
            # save weight offsets
            # save_shapeinv_unet(unet_model, save_dir)
            torch.save(unet_model.state_dict(), os.path.join(save_dir, "unet.pt"))
            # save appearanceinv encoder
            torch.save(appearance_filter_model.state_dict(), os.path.join(save_dir, "appearance_filter.pt"))
            
            loguru_info(logger, accelerator, f"[*] Weights saved at {save_dir}")
    
    
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    first_epoch = 0
    global_step = 0

    # save class embed
    domain_class_token_id = tokenizer(args.domain_class_token, add_special_tokens=False, return_tensors="pt").input_ids[0]
    shape_class_token_id = tokenizer(args.shape_class_token, add_special_tokens=False, return_tensors="pt").input_ids[0]
    appearance_class_token_id = tokenizer(args.appearance_class_token, add_special_tokens=False, return_tensors="pt").input_ids[0]
    
    loguru_info(logger, accelerator, f"domain_token: {args.domain_class_token} domain_class_token_id: {domain_class_token_id}")
    loguru_info(logger, accelerator, f"shape_token: {args.shape_class_token} shape_class_token_id: {shape_class_token_id}")
    loguru_info(logger, accelerator, f"appearance_token: {args.appearance_class_token} appearance_class_token_id: {appearance_class_token_id}")
    print(domain_class_token_id)
    assert domain_class_token_id.size(0) == 1
    assert shape_class_token_id.size(0) == 1
    assert appearance_class_token_id.size(0) == 1
    
    # get class token embedding
    class_embed = text_encoder.get_input_embeddings()(domain_class_token_id.to(accelerator.device))
    shape_embed = text_encoder.get_input_embeddings()(shape_class_token_id.to(accelerator.device))
    appearance_embed = text_encoder.get_input_embeddings()(appearance_class_token_id.to(accelerator.device))
    
    # obtain a none text embedding for features extraction
    input_ids_for_encoder = tokenizer(
        "",
        # args.prompt_template.format(placeholder_token=args.domain_class_token),
        padding="max_length", truncation=True, max_length=tokenizer.model_max_length,
        return_tensors="pt"
    ).input_ids
    
    # Get the text embedding for shapeinv conditioning
    encoder_hidden_states_for_shapeinv = text_encoder(input_ids_for_encoder.to(accelerator.device))[0].to(dtype=weight_dtype)
    
    
    # Training loop
    try:
        start_time = time.time()
        for epoch in range(first_epoch, args.num_train_epochs):
            
            student_unet.train()
            appearance_filter.train()
            
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(student_unet):
                    pixel_values = batch["pixel_values"] # 
                    # vae for image
                    latents = vae.encode(pixel_values.to(dtype=weight_dtype)).latent_dist.sample().detach()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]

                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()
                    
                    student_prompts = [f"{args.student_prompt}"] * bsz
                    
                    print(student_prompts)
                    
                    
                    # random select bsz=1 text prompt for training
                    student_prompt = random.sample(student_prompts, bsz)
                    
                    student_input_ids = tokenizer(student_prompt, padding="max_length", truncation=True, max_length=tokenizer.model_max_length,return_tensors="pt").input_ids
                    student_embeds = text_encoder.get_input_embeddings()(student_input_ids.to(accelerator.device))

                    appearance_placeholder_token_id_idxs = [i.index(placeholder_token_appearance_id) for i in student_input_ids.cpu().tolist()]

                    
                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    
                    # Get the teacher unet encoder outputs for *s
                    encoder_hidden_states_for_shapeinv_forward = encoder_hidden_states_for_shapeinv.expand(bsz, -1, -1)
                    
                    # Get the teacher unet encoder outputs for *s
                    teacher_encoder_outputs = teacher_unet(noisy_latents, timesteps, encoder_hidden_states_for_shapeinv_forward, return_encoder_outputs=True)
                    
                    # Forward E4T encoder to get the embedding for teacher
                    teacher_domain_embed = teacher_inv_encoder(x=pixel_values, unet_down_block_samples=teacher_encoder_outputs["down_block_samples"])
                    domain_embed = class_embed.clone().expand(bsz, -1) + args.domain_embed_scale * teacher_domain_embed

                    #  shape and appearance filter processing
                    appearance_embed_filter = appearance_filter(domain_embed)
                    student_appearance_embed = appearance_embed.clone().expand(bsz, -1) + args.domain_embed_scale * appearance_embed_filter
                    
                    for i, placeholder_token_id_idx in enumerate(appearance_placeholder_token_id_idxs):
                        student_embeds[i, placeholder_token_id_idx, :] = student_appearance_embed[i]
                    
                    # Get the text embedding for conditioning
                    student_encoder_hidden_states = text_encoder(inputs_embeds=student_embeds)[0].to(dtype=weight_dtype)
                    student_encoder_hidden_states = student_encoder_hidden_states.expand(bsz, -1, -1)
                    # Predict the noise residual
                    model_pred_student = student_unet(noisy_latents, timesteps, student_encoder_hidden_states).sample

                    
                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    
                
                    # appearance_k_v_loss = Self_Attn_K_V_Appearance_loss(teacher_unet=teacher_unet, student_unet=student_unet)
                    loss_diff_student = args.loss_diff_student_weight * F.mse_loss(model_pred_student.float(), target.float(), reduction="mean")
                    loss_reg_student = args.reg_lambda * student_appearance_embed.pow(2).sum()
                    # loss = loss_diff_student + loss_reg_student + appearance_k_v_loss
                    loss = loss_diff_student + loss_reg_student
                    accelerator.backward(loss)
                    

                    # if accelerator.sync_gradients:
                    #     params_to_clip = itertools.chain(unet.parameters(), e4t_encoder.parameters())
                    #     accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    # # Checks if the accelerator has performed an optimization step behind the scenes
                    # if accelerator.sync_gradients:
                    # progress_bar.update(1)
                    global_step += 1
                    if global_step % args.checkpointing_steps == 0:
                        save_weights(global_step)
                        save_path = os.path.join(weight_folder, f"checkpoint-{global_step}")
                        # accelerator.save_state(save_path)
                        # loguru_info(logger, accelerator, f"Saved state to {save_path}")

                   
                    logs = {
                        "loss": loss.detach().item(),
                        "train/loss_diff": loss_diff_student.detach().item(),
                        "train/loss_reg": loss_reg_student.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],    
                    }

                    elapsed_time, estimated_remaining_time = time_info(start_time, global_step, args.max_train_steps)
                    
                    loguru_info(logger, accelerator, f"Step {global_step}/{args.max_train_steps} | Time: {elapsed_time}/{estimated_remaining_time} | loss:{loss.detach().item():.4f}, loss_diff_student:{loss_diff_student.detach().item():.4f}, loss_reg_student:{loss_reg_student.detach().item():.4f}, lr:{lr_scheduler.get_last_lr()[0]:.3f}")
                    
                    # for wandb
                    # accelerator.log(logs, step=global_step)

                    if global_step >= args.max_train_steps:
                            break

    except KeyboardInterrupt:
        print("Summoning checkpoint...")
        pass
    accelerator.wait_for_everyone()
    save_weights(global_step)
    accelerator.end_training()
    
    
    
if __name__ == '__main__':
    main()    
