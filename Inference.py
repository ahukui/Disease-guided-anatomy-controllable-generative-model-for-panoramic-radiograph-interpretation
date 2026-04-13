#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 23:14:32 2025

@author: kui
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from datetime import datetime
from typing import List, Union, Optional
import cv2
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
from transformers import AutoTokenizer, CLIPTextModel
from model.autoencoder_kl import AutoencoderKL
from model.pipeline import StableDiffusionPipeline
from model.unet_2d_condition import UNet2DConditionModel
from model.controlnet import ControlNetModel
import json
from model.pipeline_controlnet import StableDiffusionControlNetPipeline
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer, AutoModel, CLIPTextModel
from diffusers.optimization import get_scheduler
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.training_utils import EMAModel

from Text2PR_dataset import PRGenDataset
from model.autoencoder_kl import AutoencoderKL
from model.pipeline import StableDiffusionPipeline
from model.unet_2d_condition import UNet2DConditionModel

from skimage.transform import resize

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def load_pipeline(pretrained_model_path: str, device: str = "cuda"):
    """
    Load the trained model pipeline from checkpoint
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        device: Device to run inference on
    
    Returns:
        pipeline: StableDiffusionPipeline object
    """
    print(f"Loading model from {pretrained_model_path}")
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    vae.scaling_factor = 0.18215
    
    # Load tokenizer and text encoder
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_path, 
        subfolder="tokenizer", 
        use_fast=False
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path, 
        subfolder="text_encoder"
    )
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_path, 
        subfolder="unet"
    )
    

    # Load ControlNet in float32
    controlnet = ControlNetModel.from_pretrained(
        pretrained_model_path, 
        subfolder="controlnet",
        torch_dtype=torch.float32  # Explicitly use float32
    )

    # Load scheduler
    scheduler = DDIMScheduler.from_pretrained(
        pretrained_model_path, 
        subfolder="scheduler"
    )
    
    # Create pipeline with all components in float32
    pipeline = StableDiffusionControlNetPipeline(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
    )

    pipeline.set_progress_bar_config(disable=True)

    # Move to device
    pipeline = pipeline.to(device)
    
    # Enable memory efficient attention if available
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        print("Enabled xformers memory efficient attention")
    except:
        print("xformers not available, using default attention")
    
    # Enable attention slicing for memory efficiency
    pipeline.enable_attention_slicing()
    
    return pipeline


def normalize(data):
    data = data/255.0
    return data * 2. - 1.


def select_largest_connected_region(mask, threshold=0.5):
    """
    Select only the largest connected region in a binary mask.
    """
    if len(mask.shape) == 3:
        mask = mask.squeeze(-1)
    
    binary_mask = (mask > threshold).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    
    if num_labels <= 1:
        return mask
    
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_component_mask = (labels == largest_label).astype(np.uint8)
    filtered_mask = mask * largest_component_mask
    
    return filtered_mask


def save_prompts_batch(session_dir, all_prompts, batch_num, ID):
    """Save prompts to a batch-specific file."""
    prompts_file = os.path.join(session_dir, f"Rank_{dist.get_rank()}_prompts_batch_{batch_num}_{ID}.txt")
    with open(prompts_file, "w") as f:
        for i, prompt in enumerate(all_prompts):
            f.write(f"{batch_num}_{int(ID) + int(i)}: {prompt}\n")

    print(f"[Rank {dist.get_rank()}] Saved {len(all_prompts)} prompts to {prompts_file}")

def normalize_mask(tensor: np.ndarray) -> np.ndarray:
    """Normalize mask to [0, 1] range."""
    tensor = tensor/255.0        
    return tensor* 2. - 1.

def resize_mask(m: np.ndarray) -> np.ndarray:
    """NEAREST interpolation for mask (avoid edge mixing)."""
    return resize(m, (512, 1024), order=0, preserve_range=True)

def generate_images_distributed(
    rank,
    world_size,
    pipeline,
    data_list,
    output_dir: str,
    num_images_per_prompt: int = 8,
    height: int = 256,
    width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 10,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    save_interval: int = 200,
):
    """
    Generate images from text prompts on a specific GPU
    """
    device = f"cuda:{rank}"
    
    # Set seed if provided
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed + rank)
    else:
        generator = None

    # Divide data among GPUs
    total_samples = len(data_list)
    samples_per_gpu = total_samples // world_size
    start_idx = rank * samples_per_gpu
    end_idx = start_idx + samples_per_gpu if rank < world_size - 1 else total_samples
    
    local_data = data_list[start_idx:end_idx]
    
    print(f"[Rank {rank}] Processing samples {start_idx} to {end_idx-1} ({len(local_data)} samples)")

    all_prompts = []
    batch_counter = 0

    # Generate images for each prompt
    for local_idx, prompts in enumerate(tqdm(local_data, desc=f"GPU {rank}", position=rank)):
        global_idx = local_idx# start_idx #+ local_idx local_idx#
        
        if local_idx % 100 == 0:
            print(f"[Rank {rank}] Processing sample {local_idx}/{len(local_data)}")

        text = prompts["content"]
        skeleton_path = prompts["images"]
        
        # Load skeleton
        skeleton = Image.open(skeleton_path).convert("L")
        skeleton = np.array(skeleton, dtype=np.float32)
        cv2.imwrite(
            os.path.join(output_dir, f"rank{rank}_{global_idx}_skeleton.png"), 
            skeleton
        )

        skeleton = normalize_mask(skeleton)
        skeleton = resize_mask(skeleton)
        skeleton = torch.tensor(skeleton, dtype=torch.float32).unsqueeze(0)
        #print(f"Prompt: {text}")
        all_prompts.append(text)
        
        # Generate images
        with torch.no_grad():
            images = pipeline(
                prompt=text,
                image=skeleton,                
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
            )
        #print('~~~~~~~~~~~~~~', images.shape) ###   torch.Size([8, 3, 512, 1024])
        # Save generated images
        for idx in range(images.shape[0]):
            # IM = (images[idx, 0:2, :, :] + 1.0) / 2.0   # [-1,1] → [0,1]

            # # BCHW → HWC
            # IM = IM.permute(1, 2, 0).detach().cpu().numpy()
            # IM = np.clip(IM, 0.0, 1.0)

            # # Pad to 3 channels for OpenCV
            # zero_channel = np.zeros_like(IM[:, :, :1])
            # IM = np.concatenate([IM, zero_channel], axis=2)

            # # Convert to uint8
            # IM = (IM * 255).astype(np.uint8)

            IM = (images[idx,:,:,:]+ 1.) / 2. 
            #IM = (IM + 1.) / 2. #[idx, 0:1, :, :]- images[idx, 1:2, :, :] # .squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            IM = IM.permute(1, 2, 0).detach().cpu().numpy() #squeeze(0).permute(1, 2, 0).
            IM = np.clip(IM, 0, 1)
            IM = (IM * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(output_dir, f"rank{rank}_{global_idx}_{idx}.png"), 
                IM
            )
        
        # Save prompts every save_interval samples
        if (local_idx + 1) % save_interval == 0:
            save_prompts_batch(output_dir, all_prompts, f"rank{rank}", f"{global_idx-(save_interval-1)}")
            all_prompts = []  # Clear the list
            #batch_counter += 1
    
    # Save any remaining prompts
    if all_prompts:
        save_prompts_batch(output_dir, all_prompts, batch_counter)
    
    print(f"[Rank {rank}] Completed processing {len(local_data)} samples")


def run_distributed_inference(
    rank,
    world_size,
    args,
    data_list,
    session_dir,
):
    """
    Run inference on a specific GPU
    """
    setup(rank, world_size)
    
    # Set device for this process
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    
    # Load pipeline on this GPU
    pipeline = load_pipeline(args.checkpoint_path, device)
    
    # Set scheduler if different from default
    if args.scheduler == "dpm":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        if rank == 0:
            print("Using DPM solver scheduler")
    
    # Generate images
    generate_images_distributed(
        rank=rank,
        world_size=world_size,
        pipeline=pipeline,
        data_list=data_list,
        output_dir=session_dir,
        num_images_per_prompt=args.num_images,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        save_interval=args.save_interval,
    )
    
    cleanup()


def main():
    parser = argparse.ArgumentParser(description="Generate images from trained diffusion model")
    
    # Model arguments
    parser.add_argument(
        "--checkpoint_path",
        default="./checkpoint",
        type=str,
        help="Path to model checkpoint directory"
    )
    
    # Generation arguments
    parser.add_argument(
        "--data_path",
        default="./PatientInf.json",
        type=str,
        help="Path to data JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_outputs",
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=8,
        help="Number of images to generate per prompt"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ddim",
        choices=["ddim", "dpm", "euler", "pndm"],
        help="Scheduler to use for inference"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=torch.cuda.device_count(),
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5,
        help="Save prompts every N samples"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(args.output_dir, f"generation_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    
    # Load dataset metadata
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    print(f"[INFO] Loaded {len(data_list)} samples")
    print(f"[INFO] Using {args.num_gpus} GPUs")
    print(f"[INFO] Saving prompts every {args.save_interval} samples")
    print(f"[INFO] Output directory: {session_dir}")
    
    # Launch distributed inference
    world_size = args.num_gpus
    mp.spawn(
        run_distributed_inference,
        args=(world_size, args, data_list, session_dir),
        nprocs=world_size,
        join=True
    )
    
    print(f"\nGeneration complete! All images saved to {session_dir}")


if __name__ == "__main__":
    main()