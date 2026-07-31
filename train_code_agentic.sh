#!/bin/bash
#SBATCH --job-name=LORE_TRAIN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=./%j.out
#SBATCH --error=./%j.err
#SBATCH -p debug
#SBATCH --gres=gpu:1
# #SBATCH --nodelist=hpc[1-3]

# ─────────────────────────────────────────────────────────────────────────────
# LORE / MedSA training
#   sbatch train_code_agentic.sh <dataset>
#   dataset: hecktor | colon | pancreas | lits | kits | autopet | brats
# ─────────────────────────────────────────────────────────────────────────────

# Resolve repo root (directory of this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET=${1:-hecktor}
EXTRA_ARGS=()

case "$DATASET" in
    hecktor)
        DATA_DIR=src-agentic/splits/hecktor
        SAVE_NAME=hecktor_medsa
        EXTRA_ARGS=(
            --medsa_spatial_ramp_end 100
            --medsa_rl_start_epoch 40
        )
        ;;
    colon)
        DATA_DIR=src-agentic/splits/colon
        SAVE_NAME=colon_medsa
        ;;
    pancreas)
        DATA_DIR=src-agentic/splits/pancreas
        SAVE_NAME=pancreas_medsa
        ;;
    lits)
        DATA_DIR=src-agentic/splits/lits
        SAVE_NAME=lits_medsa
        ;;
    kits)
        DATA_DIR=src-agentic/splits/kits
        SAVE_NAME=kits_medsa
        ;;
    autopet)
        DATA_DIR=src-agentic/splits/autopet
        SAVE_NAME=autopet_medsa
        EXTRA_ARGS=(
            --medsa_noise_anneal_end 100
        )
        ;;
    brats)
        DATA_DIR=src-agentic/splits/brats
        SAVE_NAME=brats_medsa
        ;;
    *)
        echo "Unknown dataset: '$DATASET'"
        echo "Choose one of: hecktor | colon | pancreas | lits | kits | autopet | brats"
        exit 1
        ;;
esac

scontrol update JobId=$SLURM_JOB_ID JobName=${DATASET}_${SAVE_NAME}

source "${LORE_ENV:-$HOME/prism_env312}/bin/activate"
python -V

LOG_DIR=./implementation_medsa/${DATASET}/${SAVE_NAME}
mkdir -p "$LOG_DIR"

if [ -f "${LOG_DIR}/best.pth.tar" ]; then
    echo "Found best.pth.tar — resuming from best checkpoint"
    EXTRA_ARGS+=(--resume --resume_best)
fi

AGENT_ARGS=(
    --use_medsa
    --spatial_sigma 5.0
    --medsa_noise_sigma_start  15.0
    --medsa_noise_anneal_end   150
    --medsa_spatial_ramp_end  80
    --medsa_rl_start_epoch    50
    --medsa_rl_ramp_len       100
    --dqn_replay_capacity     10000
    --dqn_batch_size          64
    --dqn_gamma               0.99
    --dqn_target_update_freq  100
    --medsa_lr                1e-4
)

env PYTHONPATH=$(pwd) PYTHONUNBUFFERED=1 \
    python src-agentic/train.py \
        --data          "$DATASET" \
        --save_dir      ./implementation_medsa \
        --save_name     "$SAVE_NAME" \
        --data_dir      "$DATA_DIR" \
        --num_workers   2 \
        --split         train \
        --model_type    vit_b_ori \
        --lr            4e-5 \
        --lr_scheduler  linear \
        --max_epoch     350 \
        --image_size    128 \
        --batch_size    1 \
        --checkpoint    best \
        --checkpoint_sam ./checkpoint_sam/sam_vit_b_01ec64.pth \
        --num_classes   2 \
        --tolerance     5 \
        --boundary_kernel_size 5 \
        --accumulation_steps 20 \
        --iter_nums     11 \
        --num_clicks    50 \
        --num_clicks_validation 10 \
        --use_box \
        --use_scribble \
        --num_multiple_outputs 3 \
        --multiple_outputs \
        --refine \
        --dynamic \
        --efficient_scribble \
        --device cuda:0 \
        "${AGENT_ARGS[@]}" \
        "${EXTRA_ARGS[@]}" \
        > "${LOG_DIR}/train_$(date +%Y%m%d-%H%M%S).log" 2>&1
