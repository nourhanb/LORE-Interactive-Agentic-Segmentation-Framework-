# LORE: Learning to Reason Over Physician Corrections

**Accepted at MICCAI 2026 Workshop: [CLiMeM](https://climem.github.io/CLiMeM/index.html) — Continual Learning in Medical Multimodal-Vision**

Interactive agentic framework for **3D tumor segmentation**. A policy learns *what* correction to request (click, box, scribble, …) and *when* to stop, conditioned on the current mask, error residuals, and geometric descriptors.

This repository contains the LORE / MedSA agent code (`src-agentic/`) and launch scripts. Datasets and trained weights are not included.

## Abstract

Interactive segmentation models ask a physician to iteratively correct a model’s output until the result is acceptable, yet existing methods treat each correction as an isolated reactive event with no structured reasoning about the interaction itself. In this paper, we propose LORE, an interactive agentic framework built on a frozen 3D SAM backbone that trains an agent to reason over the full history of physician corrections so that it can ask for the right kind of correction at the right time. The agent learns through the Chain of Clinical Corrections (CoCC): it observes the geometry of the current error, accumulates correction patterns across the session to detect recurring structural failures, hypothesizes where the next correction will fall, and verifies its own spatial prediction against the physician’s subsequent action via a novel spatial grounding reward, making the agent rewarded not only for segmentation quality, but for correctly anticipating where the physician will need to act next. Evaluated across six benchmarks spanning CT (colon, pancreas, liver, kidney) and FDG-PET (head-and-neck, lung), LORE consistently achieves strong segmentation quality while substantially reducing physician interaction burden.

## Repository layout

```text
src-agentic/
  train.py / test.py          # entry points
  config/                     # argparse + model/dataloader setup
  dataset/                    # TorchIO / MONAI loaders
  models/                     # SAM-3D, MedSA (LORE), policy
  processor/                  # training loop
  utils/                      # scribbles, clinical priority, logging
  voxynth/                    # scribble deformation helpers
train_code_agentic.sh         # SLURM train launcher
test_code_agentic.sh          # SLURM test launcher
setup_env312.sh               # optional env bootstrap
requirements.txt
```

## Requirements

- Python 3.12 recommended  
- CUDA GPU + PyTorch  
- SAM ViT-B weights at `./checkpoint_sam/sam_vit_b_01ec64.pth`  
  ([download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth))

### Environment

```bash
# Option A — use the helper (creates $LORE_ENV or ~/prism_env312)
sbatch setup_env312.sh
# or: bash setup_env312.sh

# Option B — existing venv
export LORE_ENV=/path/to/your/venv
source "$LORE_ENV/bin/activate"
pip install -r requirements.txt
pip install torch torchvision torchaudio   # match your CUDA
```

Launch scripts activate `${LORE_ENV:-$HOME/prism_env312}` by default.

## Data

Place each dataset under:

```text
src-agentic/splits/<dataset>/
```

Supported names: `hecktor`, `colon`, `pancreas`, `lits`, `kits`, `autopet`, `brats`.

Expected layout follows the project’s TorchIO / split conventions (images, labels, and `split.pkl` or equivalent). Paths are set in the train/test scripts via `--data_dir`.

## Train

```bash
sbatch train_code_agentic.sh <dataset>
# examples:
sbatch train_code_agentic.sh hecktor
sbatch train_code_agentic.sh colon
```

All datasets use LORE / MedSA (`--use_medsa`).

Checkpoints and logs are written to:

```text
./implementation_medsa/<dataset>/<save_name>/
```

(`best.pth.tar` is used for evaluation.)

## Test

```bash
sbatch test_code_agentic.sh <dataset>
```

Reports are written under the run directory, e.g. `test_reports/` (per-case CSV, summary JSON, Dice-vs-step curves).

## Manual invocation

```bash
export PYTHONPATH=$(pwd):$(pwd)/src-agentic

python src-agentic/train.py \
  --data hecktor \
  --data_dir src-agentic/splits/hecktor \
  --save_dir ./implementation_medsa \
  --save_name hecktor_medsa \
  --checkpoint_sam ./checkpoint_sam/sam_vit_b_01ec64.pth \
  --use_medsa \
  ...

python src-agentic/test.py \
  --data hecktor \
  --data_dir src-agentic/splits/hecktor \
  --save_dir ./implementation_medsa \
  --save_name hecktor_medsa \
  --checkpoint best \
  --checkpoint_sam ./checkpoint_sam/sam_vit_b_01ec64.pth \
  --use_medsa \
  ...
```

See `train_code_agentic.sh` / `test_code_agentic.sh` for the full flag sets used in experiments.

## Citation

If you use this code, please cite:

> *Learning to Reason Over Physician Corrections: An Interactive Agentic Framework for 3D Tumor Segmentation.*  
> MICCAI 2026 Workshop on Continual Learning in Medical Multimodal-Vision (CLiMeM).
