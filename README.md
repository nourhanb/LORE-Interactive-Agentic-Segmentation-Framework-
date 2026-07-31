# LORE — Learning to Reason Over Physician Corrections

Interactive agentic framework for **3D tumor segmentation**. A policy learns *what* correction to request (click, box, scribble, …) and *when* to stop, conditioned on the current mask, error residuals, and geometric descriptors.

This repository contains the LORE / MedSA agent code (`src-agentic/`) and launch scripts. Datasets and trained weights are not included.

## Repository layout

```text
src-agentic/
  train.py / test.py          # entry points
  config/                     # argparse + model/dataloader setup
  dataset/                    # TorchIO / MONAI loaders
  models/                     # SAM-3D, MedSA (LORE), policy, CPC–UGA
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

| Dataset | Default agent |
|---------|----------------|
| `hecktor`, `autopet`, `brats` | LORE / MedSA (`--use_medsa`) |
| `colon`, `pancreas`, `lits`, `kits` | CPC–UGA (`--use_cpc_uga`) |

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

If you use this code, please cite the LORE paper:

> *Learning to Reason Over Physician Corrections: An Interactive Agentic Framework for 3D Tumor Segmentation*
