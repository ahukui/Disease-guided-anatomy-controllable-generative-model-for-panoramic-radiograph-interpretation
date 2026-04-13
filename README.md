# Disease-guided Anatomy-controllable Generative Model for Panoramic Radiograph Interpretation

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)]()
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-76B900.svg)]()
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)]()

Source code for a **disease-guided, anatomy-controllable generative model** for panoramic radiograph (PR) interpretation.  
This project aims to generate realistic panoramic radiographs with both **disease-level semantic guidance** and **anatomical controllability**, enabling downstream applications such as radiology image synthesis, mask generation, and data augmentation for diagnostic and segmentation models.

---

## Overview

Panoramic radiograph generation is a challenging task due to the complex anatomical structures of teeth and jaws, as well as the diverse disease patterns present in clinical practice. This project introduces a multi-stage generative framework for PR synthesis that supports:

- **Latent encoding** of PR images and corresponding masks
- **Text-guided PR generation** from radiology-style prompts
- **Sketch-conditioned controllable generation** for anatomy-aware PR and mask synthesis

The full training pipeline consists of three stages:

1. **Autoencoder training** for learning latent representations of PR images and masks  
2. **Text-guided pretraining** for generating PR images from text prompts  
3. **Sketch-conditioned finetuning** for anatomy-controllable PR and mask generation  

---

## Highlights

- Disease-guided panoramic radiograph generation from text prompts
- Anatomy-controllable image synthesis with sketch or mask conditioning
- Joint generation of PR images and corresponding masks
- Multi-stage training pipeline for stable and high-quality synthesis
- Flexible framework for downstream diagnosis, segmentation, and data augmentation tasks
- Multi-GPU training support using `accelerate`

---

## Environment Setup
```bash
conda env create -f environment.yaml
conda activate PRGen
```

## Training

The complete training pipeline consists of three stages: latent representation learning, text-guided pretraining, and sketch-conditioned finetuning for anatomy-controllable panoramic radiograph generation.

### Stage 1 — Autoencoder Training for Latent Encoding

In the first stage, `train_vae_1024_512.py` is used to train a variational autoencoder (VAE) that compresses panoramic radiographs and their corresponding masks into latent representations. This stage establishes the latent space for subsequent diffusion-based generation.

**Configuration:** `./training_configs/train_vae_config_1024_512.yml`
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_vae_1024_512.py
```

### Stage 2 — Text-Guided Pretraining for Text-Derived PR Generation

In the second stage, `train_unet_1024_512.py` is used to train a diffusion U-Net for panoramic radiograph synthesis conditioned on text prompts. This stage enables the model to learn semantic alignment between radiology-style textual descriptions and image generation.

**Configuration:** `./training_configs/train_unet_config.yml`
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 12345 train_unet_1024_512.py
```

### Stage 3 — Sketch-Conditioned Finetuning for Anatomy-Controllable PR Generation

In the final stage, `train_text_mask.py` is used to finetune the diffusion model for joint PR image and mask synthesis conditioned on both text prompts and structural guidance. This stage improves anatomical controllability and supports structure-aware generation.

**Configuration:** `./training_configs/train_controlnet_config.yml`
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 12345 train_text_mask.py
```

## Inference 

Once training is complete, you can use `Inference.py` to perform multi-GPU inference for panoramic radiograph generation.

Pretrained model weights are available at [Google Drive](https://drive.google.com/drive/folders/1khFkvEWGQ0D869hOqaZgbRSjbBte9hN2?usp=drive_link).
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python your_script.py \
  --checkpoint_path ./checkpoint \
  --data_path ./PatientInf.json \
  --output_dir ./inference_outputs \
  --num_gpus 4
```
