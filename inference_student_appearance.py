import argparse
import os
from tqdm import tqdm
from PIL import Image
import torch
from diffusers import (
    LMSDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
)
from diffusers.utils import is_xformers_available
from shapeinv.encoder import ShapeInvEncoder
from shapeinv.models.modeling_clip import CLIPTextModel
from shapeinv.pipeline_stable_diffusion_e4t import StableDiffusionE4TPipeline_appearance
from shapeinv.utils import load_config_from_pretrained, load_shapeinv_encoder, load_shapeinv_unet,load_student_unet
from shapeinv.utils import load_image, AttributeDict
from transformers import CLIPTokenizer
from filter import MLP
from diffusers import StableDiffusionPipeline
import numpy as np
from torch import autocast, inference_mode
import time
import matplotlib.pyplot as plt

from ddim_custom import DDIMScheduler_for_inversion

from register import register_attention_control, register_attention_control_ostaf

from controller import Controller

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def show_lat(latents, pipe):
    # utility function for visualization of diffusion process
    with torch.no_grad():
        images = pipe.decode_latents(latents)
        im = pipe.numpy_to_pil(images)[0].resize((128, 128))
    return im

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path_or_url", type=str, help="path to the input image")
    parser.add_argument("--pretrained_model_name_or_path", type=str, help="model dir including config.json, encoder.pt, weight_offsets.pt")
    # diffusers config
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="unconditional guidance scale")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space",)
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="number of images per prompt")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="number of ddim sampling steps")
    parser.add_argument("--prompt", type=str, nargs="?", default="a photo of *s", help="the prompt to render")
    parser.add_argument("--scheduler_type", type=str, choices=["ddim", "plms", "lms", "euler", "euler_ancestral", "dpm_solver++"], default="ddim", help="diffusion scheduler type")
    parser.add_argument("--seed", type=int, default=None, help="the seed (for reproducible sampling)")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space",)
    parser.add_argument("--target_folder", type=str, help="path to the output folder")
    # new add
    parser.add_argument("--teacher_model_name_or_path", type=str, help="teacher model dir including config.json, encoder.pt, unet.pt")
    parser.add_argument("--edit_image_path_or_url", type=str)
    parser.add_argument("--source_prompt", type=str, default="a photo of *s", help="the prompt to edit")
    parser.add_argument("--edit_mask", type=str)
    parser.add_argument("--time", type=int)
    parser.add_argument("--img_pre_name", type=str)
    
    
    opt = parser.parse_args()
    return opt


def getattr_from_config(config, key):
    if config.pretrained_args is not None:
        return config.pretrained_args[key]
    else:
        # pre-training phase model
        value = getattr(config, key)
        assert value is not None
        return value


def get_e4t_config(config):
    return AttributeDict(config.pretrained_args) if config.pretrained_args is not None else config


SCHEDULER_MAPPING = {
    "ddim": DDIMScheduler,
    "plms": PNDMScheduler,
    "lms": LMSDiscreteScheduler,
    "euler": EulerDiscreteScheduler,
    "euler_ancestral": EulerAncestralDiscreteScheduler,
    "dpm_solver++": DPMSolverMultistepScheduler,
}


def ddim_inversion(pretrained_model_name_or_path,device,args):
    
    # sd-unet 
    ddim_pipe = StableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path,
                scheduler=DDIMScheduler_for_inversion(beta_end=0.012,
                                        beta_schedule="scaled_linear",
                                        beta_start=0.00085)
                ).to(device)
    
    
    ddim_UNet = ddim_pipe.unet
    ddim_UNet.to(device)
    
    def preprocess(image):
        w, h = image.size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        image = image.to(device)
        return 2.0 * image - 1.0  # normalize to [-1, 1]`

    def im2latent(pipe, im, generator):
        init_image = preprocess(im).to(pipe.device)
        init_latent_dist = pipe.vae.encode(init_image).latent_dist
        init_latents = init_latent_dist.sample(generator=generator)
        
        return init_latents * 0.18215
    
    # ddim_pipe = pipe
    init_image = Image.open(f"{args.edit_image_path_or_url}").convert("RGB").resize((512, 512))
    g = torch.Generator(device=ddim_pipe.device).manual_seed(args.seed)
    image_latents = im2latent(ddim_pipe, init_image, g)
    ddim_pipe.scheduler.set_timesteps(51)
    source_prompt = f"{args.source_prompt}"
    context = ddim_pipe._encode_prompt(source_prompt, ddim_pipe.device, 1, False, "")
    decoded_latents_sd = image_latents.clone().to(device)
    
    with autocast("cuda"), inference_mode():
        # we are pivoting timesteps as we are moving in opposite direction
        timesteps = ddim_pipe.scheduler.timesteps.flip(0)
        # this would be our targets for pivoting
        init_trajectory = torch.empty(len(timesteps), *decoded_latents_sd.size()[1:], device=decoded_latents_sd.device, dtype=decoded_latents_sd.dtype)
        for i, t in enumerate(tqdm(timesteps)):
            init_trajectory[i:i+1] = decoded_latents_sd
            noise_pred = ddim_pipe.unet(decoded_latents_sd, t, encoder_hidden_states=context).sample
            decoded_latents_sd = ddim_pipe.scheduler.reverse_step(noise_pred, t, decoded_latents_sd).next_sample
        
    return decoded_latents_sd,ddim_pipe,context


def main():
    
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    
    pretrained_folder = os.path.join(args.pretrained_model_name_or_path, str(100))
    print(f"Loading from {pretrained_folder}")
    config = load_config_from_pretrained(pretrained_folder)
    
    pretrained_model_name_or_path = getattr_from_config(config, "pretrained_model_name_or_path")
    
    
    decoded_latents_sd,ddim_pipe,context = ddim_inversion(pretrained_model_name_or_path,device,args)
    controller = Controller()
    register_attention_control(ddim_pipe,controller)
    controller.set_ddl_time = args.time
    
    plt.figure(figsize=(20,8))
    latents = decoded_latents_sd.clone()
    # import ipdb; ipdb.set_trace()
    with autocast("cuda"), inference_mode():
        for i, t in enumerate(tqdm(ddim_pipe.scheduler.timesteps)):
            controller.set_cur_step(i)
            latents = ddim_pipe.scheduler.step(
                ddim_pipe.unet(latents, t, encoder_hidden_states=context).sample, t, latents
            ).prev_sample
            
            if i % 10 == 0:
                plt.subplot(1,6,i//10+1)
                plt.imshow(show_lat(latents, ddim_pipe))
    
    plt.savefig("latents.png")
    
    
    # text encoder
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    e4t_config = get_e4t_config(config)
    num_added_tokens = tokenizer.add_tokens(e4t_config.placeholder_token)
    num_added_tokens_shape = tokenizer.add_tokens(e4t_config.placeholder_token_shape)
    num_added_tokens_appearance = tokenizer.add_tokens(e4t_config.placeholder_token_appearance)
    if num_added_tokens == 0 or num_added_tokens_shape == 0 or num_added_tokens_appearance == 0:
        raise ValueError(
            f"The tokenizer already contains the token {e4t_config.placeholder_token} or {e4t_config.placeholder_token_shape} or {e4t_config.placeholder_token_appearance}. Please pass a different `placeholder_token` that is not already in the tokenizer.")
    text_encoder.resize_token_embeddings(len(tokenizer))
    
    
    # unet
    # * 加载teacher model
    teacher_unet = load_shapeinv_unet(
        pretrained_model_name_or_path
        # ckpt_path=os.path.join(args.teacher_model_name_or_path, "unet.pt"),
    )
    # e4t encoder
    teacher_encoder = load_shapeinv_encoder(
        # ckpt_path=args.teacher_model_name_or_path,
        word_embedding_dim=text_encoder.config.hidden_size,
        clip_model=getattr_from_config(config, "pretrained_model_name_or_path")
    )
    
    for weight in range(1000,20000,100):
        
        student_unet_pretrained_path = os.path.join(args.pretrained_model_name_or_path, str(weight))
        student_unet = load_shapeinv_unet(
            ckpt_path=os.path.join(student_unet_pretrained_path, "unet.pt"),
        )
        student_unet.to(device)
        
        controller.self_idx = 0
        controller.cross_idx = 0
        controller.cur_step = 0 
        
        register_attention_control_ostaf(student_unet,controller,args.time)
        
        appearance_filter = MLP(input_size=768, hidden_size=768, output_size=768)
        appearance_filter.load_state_dict(torch.load(os.path.join(student_unet_pretrained_path, "appearance_filter.pt")), strict=True)
        appearance_filter.to(device)
        
        # vit_dino = VitExtractor(model_name="dino_vitb8", device=device)
        # load pipe
        pipe = StableDiffusionE4TPipeline_appearance.from_pretrained(
            pretrained_model_name_or_path,
            unet=teacher_unet,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            e4t_encoder=teacher_encoder,
            e4t_config=e4t_config,
            # scheduler=DDIMScheduler_for_inversion(beta_end=0.012,
            #                                 beta_schedule="scaled_linear",
            #                                 beta_start=0.00085),
            scheduler=SCHEDULER_MAPPING[args.scheduler_type].from_pretrained(pretrained_model_name_or_path, subfolder="scheduler"),
            requires_safety_checker=False,
            safety_checker=None,
            feature_extractor=None,
            already_added_placeholder_token=True,        
        )
        
        
        if args.enable_xformers_memory_efficient_attention:
            assert is_xformers_available()
            pipe.enable_xformers_memory_efficient_attention()
            print("Using xformers!")
        pipe = pipe.to(device)
        print("loaded pipeline")

        if os.path.exists(os.path.join(student_unet_pretrained_path, "text_encoder.pt")):
            ckpt_path = os.path.join(student_unet_pretrained_path, "text_encoder.pt")
            state_dict = torch.load(ckpt_path, map_location="cpu")
            print(f"Resuming from {ckpt_path}")
            m, u = text_encoder.load_state_dict(state_dict, strict=False)
            if len(m) > 0:
                raise RuntimeError(f"missing keys:\n{m}")
            if len(u) > 0:
                raise RuntimeError(f"unexpected keys:\n{u}")
        
        
        
        
        
        
        
        # * DDIM Inversion Reconstructing Ending
        
        # bkg_latents = (decoded_latents.clone() + decoded_latents_sd.clone())/2
        bkg_latents = decoded_latents_sd.clone()
        
        register_attention_control(ddim_pipe,controller,flag=False)
        
        # run!
        # download an image
        image = load_image(args.image_path_or_url)
        
        generator = None
        if args.seed:
            print("使用了seed！")
            generator = torch.Generator(device=device).manual_seed(args.seed)
            # generator = torch.cuda.manual_seed_all(args.seed)
        
        # switch to infer mode
        with torch.no_grad():
        # import ipdb;ipdb.set_trace()    
            prompts = args.prompt.split("::")
            all_images = []
            for prompt in tqdm(prompts):
                with torch.autocast(device), torch.inference_mode():
                    images = pipe(
                        prompt,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=generator,
                        image=image,
                        num_images_per_prompt=args.num_images_per_prompt,
                        height=args.height,
                        width=args.width,
                        student_unet=student_unet,
                        appearance_filter=appearance_filter,
                        dino_vit=None,
                        # sd_unet = ddim_UNet,
                        bkg_latents=bkg_latents,
                        ddim_pipe=ddim_pipe,
                        context_ddim=context,
                        mask_path=args.edit_mask,
                        controller=controller,
                    ).images
                all_images.extend(images)
            grid_image = image_grid(all_images, len(prompts), args.num_images_per_prompt)
            time_str = str(time.time())
            
            
            
            os.makedirs(args.target_folder, exist_ok=True)
            grid_image.save(args.target_folder + str(args.time)+ "_" + str(weight) + ".png")
            print("DONE! See `grid.png` for the results!")


if __name__ == '__main__':
    main()

