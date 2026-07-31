#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# LORE / MedSA test — runs src-agentic/test.py on the test split.
#
# Reports under ${SAVE_DIR}/test_reports/:
#   • per-case CSV, summary JSON, Dice-vs-step curves
#
# Usage:
#   sbatch test_code_agentic.sh <dataset>
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=LORE_TEST
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=./%j.out
#SBATCH --error=./%j.err
#SBATCH -p debug
#SBATCH --gres=gpu:1
# #SBATCH --nodelist=hpc[1-3]

# Resolve repo root (directory of this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET=${1:-hecktor}

source "${LORE_ENV:-$HOME/prism_env312}/bin/activate"
python -V

case "$DATASET" in
    hecktor)  DATA_DIR=src-agentic/splits/hecktor;  SAVE_NAME=hecktor_medsa ;;
    colon)    DATA_DIR=src-agentic/splits/colon;    SAVE_NAME=colon_medsa ;;
    pancreas) DATA_DIR=src-agentic/splits/pancreas; SAVE_NAME=pancreas_medsa ;;
    lits)     DATA_DIR=src-agentic/splits/lits;     SAVE_NAME=lits_medsa ;;
    kits)     DATA_DIR=src-agentic/splits/kits;     SAVE_NAME=kits_medsa ;;
    autopet)  DATA_DIR=src-agentic/splits/autopet;  SAVE_NAME=autopet_medsa ;;
    brats)    DATA_DIR=src-agentic/splits/brats;    SAVE_NAME=brats_medsa ;;
    *)
        echo "Unknown dataset: $DATASET"
        exit 1
        ;;
esac

SAVE_ROOT=./implementation_medsa
SAVE_DIR=${SAVE_ROOT}/${DATASET}/${SAVE_NAME}
LOG_DIR=${SAVE_DIR}
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="${LOG_DIR}/test_${SAVE_NAME}_${TIMESTAMP}.log"

CKPT_FLAG=best
if [ ! -f "${SAVE_DIR}/best.pth.tar" ]; then
    if [ -f "${SAVE_DIR}/last.pth.tar" ]; then
        echo "WARNING: best.pth.tar not found; using last.pth.tar"
        CKPT_FLAG=last
    else
        echo "ERROR: no checkpoint in ${SAVE_DIR}"
        exit 1
    fi
fi

echo "Testing LORE on dataset: $DATASET"
echo "Checkpoint: ${SAVE_DIR}/${CKPT_FLAG}.pth.tar"
echo "Log: $LOG_FILE"
echo "Reports → ${SAVE_DIR}/test_reports/"

export PYTHONPATH=$(pwd):$(pwd)/src-agentic
python src-agentic/test.py \
        --data              "$DATASET" \
        --save_dir          "$SAVE_ROOT" \
        --save_name         "$SAVE_NAME" \
        --data_dir          "$DATA_DIR" \
        --split             test \
        --checkpoint        "$CKPT_FLAG" \
        --model_type        vit_b_ori \
        --checkpoint_sam    ./checkpoint_sam/sam_vit_b_01ec64.pth \
        --num_classes       2 \
        --image_size        128 \
        --iter_nums         11 \
        --num_clicks        50 \
        --num_clicks_validation 10 \
        --use_box \
        --use_scribble \
        --num_multiple_outputs 3 \
        --multiple_outputs \
        --refine \
        --dynamic \
        --efficient_scribble \
        --save_csv \
        --use_medsa \
        --device            cuda:0 \
        2>&1 | tee "$LOG_FILE"

echo "Done. See log + ${SAVE_DIR}/test_reports/"
