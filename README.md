# Restoration Score Distillation (RSD)

**[ICLR 2026 Poster]** Official PyTorch implementation of:

> **Score Distillation Beyond Acceleration: Generative Modeling from Corrupted Data**  
> [Yasi Zhang](https://yasminzhang.github.io/)<sup>†</sup>, [Tianyu Chen](https://tianyucodings.github.io/)<sup>†</sup>, Zhendong Wang, Ying Nian Wu, Mingyuan Zhou, [Oscar Leong](https://www.oscarleong.com/)  
> [arXiv:2505.13377](https://arxiv.org/abs/2505.13377) | [OpenReview](https://openreview.net/forum?id=ROGCckKICU)

---

## Overview

RSD is a two-stage framework for training efficient one-step generative models **without any clean training data**. Given only corrupted observations (e.g., noisy images), RSD:

1. **Stage 1 — Teacher Pretraining:** Trains a corruption-aware diffusion model directly on the degraded measurements.
2. **Stage 2 — Score Distillation:** Distills the teacher into a one-step generator by aligning score functions via a Fisher divergence loss, achieving both faster inference and improved sample quality.

RSD generalizes to a broad class of forward operators including Gaussian denoising, random inpainting, super-resolution, and MRI reconstruction.

---

## Installation

```bash
conda env create -f environment.yml
conda activate rsd
```

---

## Repository Structure

```
RSD/
├── train.py              # Stage 1: pretrain corruption-aware teacher
├── rsd_train.py          # Stage 2: RSD distillation into one-step generator
├── scripts/              # Utility scripts
│   ├── generate.py           # EDM sampling from teacher
│   ├── rsd_generate.py       # One-step generation from RSD model
│   ├── rsd_generate_onestep.py
│   ├── eval_fid.py           # FID evaluation
│   ├── fid.py
│   ├── dataset_tool.py       # Dataset preprocessing
│   └── rsd_metrics.py        # FID/IS/Precision/Recall metrics
├── run_bash/             # Training and evaluation scripts
│   ├── pretrain.sh           # Stage 1 (all datasets)
│   ├── distill.sh            # Stage 2 (all datasets)
│   ├── inference.sh          # Generate from teacher
│   ├── evaluate.sh           # Compute FID
│   └── generate_rsd.sh       # One-step generation
├── training/             # Core training modules
├── metrics/              # Evaluation metrics
├── ambient_utils/        # Dataset and utility functions
├── torch_utils/          # Distributed training utilities
└── dnnlib/
```

---

## Dataset Preparation

**Option 1** - download from [yasiz/vision_data](https://huggingface.co/yasiz/vision_data/tree/main)


**Option 2** - Use `scripts/dataset_tool.py` to convert datasets into ZIP format at the desired resolution.

**FFHQ (64×64)**
```bash
python scripts/dataset_tool.py --source=/path/to/ffhq/images \
    --dest=/path/to/ffhq-64x64.zip --resolution=64x64
```

**CelebA-HQ (64×64)**
```bash
python scripts/dataset_tool.py --source=/path/to/celeba_hq/images \
    --dest=/path/to/celeba_hq-64x64.zip --resolution=64x64
```

**AFHQ-v2 (64×64)**
```bash
python scripts/dataset_tool.py --source=/path/to/afhqv2/images \
    --dest=/path/to/afhqv2-64x64.zip --resolution=64x64
```

**CIFAR-10 (32×32)**
```bash
python scripts/dataset_tool.py --source=cifar10 \
    --dest=/path/to/cifar10-32x32.zip
```

FID reference statistics can be downloaded from the [EDM release](https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/).

---

## Checkpoints download

Pretrained checkpoints for noisy image generation (Gaussian noise, σ=0.2) are released as part of the [Denoising Score Distillation collection](https://huggingface.co/collections/yasiz/denoising-score-distillation-ckpts) on Hugging Face.

| Dataset | Stage 1 — Teacher | Stage 2 — RSD (one-step) |
|---------|-------------------|--------------------------|
| FFHQ (64×64) | [yasiz/pretrain_corrupt_ffhq_sigma_0.2](https://huggingface.co/yasiz/pretrain_corrupt_ffhq_sigma_0.2) | [yasiz/distill_corrupt_ffhq_sigma_0.2](https://huggingface.co/yasiz/distill_corrupt_ffhq_sigma_0.2) |
| CelebA-HQ (64×64) | [yasiz/pretrain_corrupt_celebahq_sigma_0.2](https://huggingface.co/yasiz/pretrain_corrupt_celebahq_sigma_0.2) | [yasiz/distill_corrupt_celebahq_sigma_0.2](https://huggingface.co/yasiz/distill_corrupt_celebahq_sigma_0.2) |
| AFHQ-v2 (64×64) | [yasiz/pretrain_corrupt_afhqv2_sigma_0.2](https://huggingface.co/yasiz/pretrain_corrupt_afhqv2_sigma_0.2) | [yasiz/distill_corrupt_afhqv2_sigma_0.2](https://huggingface.co/yasiz/distill_corrupt_afhqv2_sigma_0.2) |

Download a single checkpoint with the Hugging Face CLI:

```bash
huggingface-cli download yasiz/distill_corrupt_ffhq_sigma_0.2 --local-dir ckpts/ffhq_rsd
```

Use the Stage 1 `.pkl` as `<teacher_ckpt>` for `run_bash/distill.sh`, and the Stage 2 `.pkl` with `run_bash/generate_rsd.sh` for one-step generation.

---

## Training

### Weights & Biases

Training scripts in `run_bash/`, `mri/`, `operator/`, and `inpainting/` can log to [Weights & Biases](https://wandb.ai). Set your API key before launching (do not hardcode keys in source files):

```bash
export WANDB_API_KEY=your_api_key_here
# or persist locally:
wandb login
```

Cluster launch scripts (e.g. `inpainting/run_bash/distill_cluster.sh`) expect `WANDB_API_KEY` in the environment; add `wandb login $WANDB_API_KEY` to your job setup if needed.

Before running, set the dataset paths at the top of `run_bash/pretrain.sh` and `run_bash/distill.sh`.

**Recommended**: Check out our paper to see detailed hyperparameters. For most of the settings, we use the default [EDM](https://github.com/NVlabs/edm)   configs for pretraining, and [SiD](https://github.com/mingyuanzhou/sid) configs for distillation without any modification. 

### Stage 1: Teacher Pretraining

```bash
bash run_bash/pretrain.sh <dataset> [num_gpus]
# dataset: ffhq | celeba | afhq | cifar10
```

Examples:
```bash
bash run_bash/pretrain.sh ffhq 8
bash run_bash/pretrain.sh celeba 8
bash run_bash/pretrain.sh afhq 4
bash run_bash/pretrain.sh cifar10 4
```

Key parameters in `train.py`:
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--sigma` | Gaussian noise std added to images | `0.2` |
| `--corruption_probability` | Fraction of images corrupted | `1.0` |
| `--dataset_keep_percentage` | Use a subset of the dataset | `1.0` |

### Stage 2: RSD Distillation

Pass the pretrained teacher checkpoint as the second argument.

```bash
bash run_bash/distill.sh <dataset> <teacher_ckpt> [num_gpus]
```

Examples:
```bash
bash run_bash/distill.sh ffhq ffhq/ffhq_pretrain/.../network-snapshot-XXXXXX.pkl 8
bash run_bash/distill.sh celeba celebahq/celebahq_pretrain/.../network-snapshot-XXXXXX.pkl 4
bash run_bash/distill.sh afhq afhq/afhq_pretrain/.../network-snapshot-XXXXXX.pkl 8
bash run_bash/distill.sh cifar10 cifar10/cifar_pretrain/.../network-snapshot-XXXXXX.pkl 4
```

Key parameters in `rsd_train.py`:
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--alpha` | L2 − α·L1 loss weighting | `1.2` |
| `--tmax` | Maximum reverse diffusion step | `800` |
| `--init_sigma` | Fixed noise level during distillation | `2.5` |

---

## Inference & Evaluation

**Generate from teacher (truncated EDM sampling)**
```bash
bash run_bash/inference.sh <network_pkl> <outdir> [num_gpus]
```

**One-step generation from distilled RSD model**
```bash
bash run_bash/generate_rsd.sh <network_pkl> [outdir]
```

**Compute FID**
```bash
bash run_bash/evaluate.sh <gen_path> <ref_stats.npz> [out.json] [num_gpus]
```

For example, the end-to-end pipeline for AFHQ-v2 (one-step generation + FID) on 8 GPUs, run from the repository root:

```bash
# Inputs
CKPT=ckpts/afhqv2_rsd.pkl                  # distilled RSD generator (Stage 2 .pkl)
REF_NPZ=data/afhqv2-64x64.npz              # FID reference stats
OUTDIR=outputs/afhq_rsd_eval               # 50K PNGs land here
FID_JSON=outputs/afhq_rsd_fid.json

NUM_GPUS=8
NUM_SAMPLES=50000

mkdir -p "$(dirname "$OUTDIR")"

# Step 1 — one-step generation (sigma_G=2.5 matches --init_sigma from distill.sh)
python -m torch.distributed.run --standalone --nproc_per_node=$NUM_GPUS \
    scripts/rsd_generate_onestep.py \
    --network=$CKPT \
    --outdir=$OUTDIR \
    --seeds=0-$((NUM_SAMPLES-1)) \
    --batch=64 \
    --num=$NUM_SAMPLES \
    --sigma_G=2.5

# Step 2 — FID + Inception Score (pass the PNG directory, not the .npz)
python scripts/eval_fid.py \
    --gen_path=$OUTDIR \
    --ref_stats=$REF_NPZ \
    --out_path=$FID_JSON \
    --batch_size=128
cat $FID_JSON
```

On an 8× A800 node this completes in ~38 s for generation and ~2 min for FID, yielding `FID ≈ 5.39`, `IS ≈ 8.02` for the released AFHQ-v2 σ=0.2 checkpoint. If the default `--master_port=29500` is in use, append `--master_port=<free port>` to the `torch.distributed.run` command.



## MRI experiments 

Please check out the folder `mri` and [MRI_README.md](mri/MRI_README.md)


## Deblurring + Super-resolution (with EDM's achitecture)

Please checkout the folder `operator`. The training scripts are in `operator/A_run_bash`


## Inpainting (with ambient diffusion's architecture)

Please check out the folder `inpainting`. The pretrained/teacher ckpts are available in Ambient diffusion's codebase, or simple use
```bash
wget https://zenodo.org/record/7964925/files/checkpoints.zip?download=1
```
For distillation, use the scripts in `./inpainting/run_bash`



## TODO

- [x] Release pretrained checkpoints for noisy CelebA-HQ, FFHQ, and AFHQ
- [x] test inference code script
- [x] test training code script
- [ ] Release code and checkpoints for general operator (deblurring, super-resolution)
- [ ] Release code and checkpoints for random masking operator (inpainting)
- [ ] Release code and checkpoints for MRI operator (FastMRI reconstruction)

---

## Citation

```bibtex
@inproceedings{Zhang2026Score,
  title={Score Distillation Beyond Acceleration: Generative Modeling from Corrupted Data},
  author={Zhang, Yasi and Chen, Tianyu and Wang, Zhendong and Wu, Ying Nian and Zhou, Mingyuan and Leong, Oscar},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

---

## Acknowledgements

This codebase builds upon [EDM](https://github.com/NVlabs/edm) (Karras et al., 2022) and [SiD](https://github.com/mingyuanzhou/sid) (Zhou et al., 2024). We thank the authors for releasing their code.
