# Mitigating Sexual Content Generation via Embedding Distortion in Text-conditioned Diffusion Models

**Official implementation of the paper accepted at NeurIPS 2025**

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue)](https://neurips.cc)
[![arXiv](https://img.shields.io/badge/arXiv-2501.18877-b31b1b.svg)](https://arxiv.org/abs/2501.18877v1)

> **Authors**: Jaesin Ahn, Heechul Jung (Kyungpook National University)

## Overview

Diffusion models show remarkable image generation performance following text prompts, but risk generating sexual contents. Existing approaches struggle to defend against adversarial attacks while maintaining benign image quality. We propose **DES (Distorting Embedding Space)**, a text encoder-based defense mechanism that effectively tackles these issues through innovative embedding space control.

DES transforms unsafe embeddings extracted from text encoders toward carefully calculated safe embedding regions to prevent unsafe content generation, while reproducing the original safe embeddings. DES also neutralizes the "nudity" embedding by aligning it with neutral embedding to enhance robustness against adversarial attacks.

### Key Results

- **State-of-the-Art Defense**: ASR of **9.47%** on FLUX.1 and **0.52%** on SD v1.5
- **Superior Performance**: 76.5% and 63.9% ASR reduction compared to previous SOTA methods
- **Quality Preservation**: Maintains FID and CLIP scores comparable to original models
- **Practical Efficiency**: **90 seconds training time** with **zero inference overhead**

### Supported Models

- **Stable Diffusion**: v1.4, v1.5, v2.x, XL, v3.x, v3.5
- **FLUX.1**: FLUX.1-dev

## Repository Structure

```
neurips/
├── train_des.py                   # Main DES training script
├── save_codebook.py               # Safe embedding codebook generation
├── generate.py                    # Image generation
├── fid.py                         # FID calculation
├── clipscore.py                   # CLIP Score calculation
│
├── train.sh                       # Training example script
├── generate.sh                    # Generation example script
├── evaluate.sh                    # Evaluation example script
│
├── datasets/                      # Prompt datasets (safe/unsafe prompts, attack datasets)
│
├── tasks/                         # Evaluation utilities
│   ├── img_batch_classify.py     # NudeNet classification
│   ├── img_batch_classify_q16.py # Q16 classification
│   └── utils/                    # Evaluation metrics and utilities
│
└── README.md                      # This file
```

---

## Installation

### Requirements

```bash
# Clone the repository
git clone https://github.com/amoeba04/des-neurips2025.git
cd des-neurips2025/neurips

# Create conda environment
conda create -n des python=3.8
conda activate des

# Install PyTorch 2.2.1 with CUDA 11.8 support
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install -r requirements.txt

# Install evaluation dependencies
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/boomb0om/text2image-benchmark
```

---

## Quick Start

### 1. Prepare Safe Embedding Codebook

```bash
python save_codebook.py \
    --model_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --device cuda:0 \
    --csv_path datasets/safe_prompts_copro_sexual.csv \
    --save_dir codebook_copro_sexual
```

### 2. Train DES

```bash
python train_des.py \
    --model_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --device cuda:0 \
    --codebook_dir codebook_copro_sexual \
    --unsafe_csv_path datasets/unsafe_prompts_copro_sexual.csv \
    --safe_csv_path datasets/safe_prompts_copro_sexual.csv \
    --output_dir checkpoints/des_copro_sexual \
    --num_epochs 2 \
    --learning_rate 1e-5 \
    --batch_size 128 \
    --lambda_safe 0.3 \
    --save_every 1 \
    --ablation 1 2 3 \
    --concept_prompt "nudity" \
    --concept_guidance_scale 200.0 \
    --safe_embedding_path checkpoints/des_copro_sexual/safe_embeddings.pth
```

**Training completes in ~90 seconds** for CLIP-L/14.

### 3. Generate Images

```bash
python generate.py \
    --model_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --device cuda:0 \
    --prompts_csv "datasets/coco_prompts.csv" \
    --output_path "t2i_coco" \
    --start_idx 0 \
    --end_idx 10000 \
    --training_method des \
    --text_encoder_path "checkpoints/des_copro_sexual/1e-05_0.3/checkpoint-2.pt"
```

### 4. Evaluation

#### Attack Success Rate (ASR)

```bash
# NudeNet-based evaluation
python tasks/img_batch_classify.py \
    --job nudity \
    --cls_class nudity \
    --folder_dir results/des/stable-diffusion-v1-5/t2i_sneaky/ \
    --devices 0

# Q16-based evaluation
python tasks/img_batch_classify_q16.py \
    --job nudity \
    --cls_class nudity \
    --folder_dir results/des/stable-diffusion-v1-5/t2i_sneaky/ \
    --devices 0
```

#### Image Quality Metrics

```bash
# FID (Fréchet Inception Distance)
python fid.py \
    --gen_imgs_path results/des/stable-diffusion-v1-5/t2i_coco/ \
    --coco_imgs_path datasets/coco_10k/ \
    --device cuda:0

# CLIP Score
python clipscore.py \
    --image_folder results/des/stable-diffusion-v1-5/t2i_coco/ \
    --csv_file datasets/coco_prompts.csv \
    --device cuda:0
```

---

## Usage Guide

### Training DES

#### Training Arguments

**Core Arguments:**
- `--model_path`: HuggingFace model ID or local path
- `--codebook_dir`: Safe embedding codebook directory
- `--unsafe_csv_path`: CSV file with unsafe prompts (for UEN loss)
- `--safe_csv_path`: CSV file with safe prompts (for SEP loss)
- `--output_dir`: Checkpoint save directory

**Hyperparameters (Recommended):**
- `--lambda_safe`: Balance parameter (default: 0.3)
- `--concept_guidance_scale`: Nudity subtraction scale (default: 200.0)

**Training Configuration:**
- `--num_epochs`: Training epochs (default: 2, typically sufficient)
- `--learning_rate`: Learning rate (default: 1e-5)
- `--batch_size`: 128 for SD v1.x, 16 for FLUX

#### Multi-Model Support

**SD v3.5 (Multiple Text Encoders):**

SD v3.5 has 3 text encoders:
- Encoder 1: CLIP ViT-L (768 dim)
- Encoder 2: CLIP ViT-bigG (1280 dim)
- Encoder 3: T5-XXL (4096 dim)

**Step 1: Generate codebook for specific encoder**
```bash
python save_codebook.py \
    --model_path stabilityai/stable-diffusion-3.5-medium \
    --device cuda:0 \
    --csv_path datasets/safe_prompts_copro_sexual.csv \
    --save_dir codebook_sdv3_copro_sexual \
    --encoder_idx 3  # 1, 2, or 3
```

**Step 2: Train DES on specific encoder**

For **CLIP encoders (1 or 2)** - Single GPU is sufficient:
```bash
python train_des.py \
    --model_path stabilityai/stable-diffusion-3.5-medium \
    --codebook_dir codebook_sdv3_copro_sexual \
    --unsafe_csv_path datasets/unsafe_prompts_copro_sexual.csv \
    --safe_csv_path datasets/safe_prompts_copro_sexual.csv \
    --output_dir checkpoints/des_sdv3_encoder1 \
    --text_encoder_idx 1 \
    --batch_size 128 \
    --learning_rate 1e-5 \
    --lambda_safe 0.3 \
    ...
```

For **T5 encoder (3)** - Multi-GPU recommended for 40GB GPUs:
```bash
# T5-XXL requires multi-GPU setup for 40GB GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_des.py \
    --model_path stabilityai/stable-diffusion-3.5-medium \
    --codebook_dir codebook_sdv3_copro_sexual \
    --unsafe_csv_path datasets/unsafe_prompts_copro_sexual.csv \
    --safe_csv_path datasets/safe_prompts_copro_sexual.csv \
    --output_dir checkpoints/des_sdv3_encoder3 \
    --text_encoder_idx 3 \
    --multi_gpu \
    --gradient_checkpointing \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --lambda_safe 0.3 \
    --concept_prompt "nudity" \
    --concept_guidance_scale 200.0 \
    --safe_embedding_path checkpoints/des_sdv3_encoder3/safe_embeddings.pth
```

> **Note**: T5-XXL requires significant GPU memory. With 40GB GPUs, use `--multi_gpu` flag to distribute the model across multiple GPUs automatically.

## Model Checkpoints

### Pre-trained DES Checkpoints

#### Sexual:
- CLIP-L/14 (Stable Diffusion v1.x, Stable Diffusion v3.x, FLUX.1): [link](https://www.dropbox.com/scl/fi/mjvrq2oynmawvjbkkxpau/des.pt?rlkey=vb2bnt8fh8zm3qpbssoq7fziw&st=9c364ns3&dl=0)
- CLIP-G/14 (Stable Diffusion v3.x): [link](https://www.dropbox.com/scl/fi/qlmzjv33tmivnpu1qv5sa/des_clipg.pt?rlkey=2feesjw0qrk3c4vgo2el7foao&st=9qtfzfbi&dl=0)
- T5-XXL (Stable Diffusion v3.x, FLUX.1): [link](https://www.dropbox.com/scl/fi/sofnetmz780jnnxqf3hr0/des_t5.pt?rlkey=3lpreqkc8nuggvezbfeyekgwz&st=yrpdxa8a&dl=0)

#### Sexual, Violence, Illegal:
- CLIP-L/14 (Stable Diffusion v1.x): [link](https://www.dropbox.com/scl/fi/yvtdbuwmtlskfqh8mefil/des_multi.pt?rlkey=vwnk46lc7rdfapc7panope0hj&st=r3ujh1ne&dl=0)

#### Van Gogh:
- CLIP-L/14 (Stable Diffusion v1.x): [link](https://www.dropbox.com/scl/fi/fk5xbd8u06szf2a1por7e/des_vangogh.pt?rlkey=lxkjk67a5336twz25oju846sx&st=ugbxo5bf&dl=0)
---

## License

This project is licensed under the **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International License).

For detailed terms and conditions, see the [LICENSE](LICENSE) file.

---

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@inproceedings{ahn2025des,
  title={Mitigating Sexual Content Generation via Embedding Distortion in Text-conditioned Diffusion Models},
  author={Ahn, Jaesin and Jung, Heechul},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```
