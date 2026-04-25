#!/bin/bash
#BSUB -J bioreason_kegg_sft
#BSUB -P acc_genome_foundation
#BSUB -q gpu
#BSUB -n 8
#BSUB -R "rusage[mem=24000]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -W 12:00
#BSUB -o /sc/arion/work/cardia04/BioReason/logs/kegg_sft_%J.out
#BSUB -eo /sc/arion/work/cardia04/BioReason/logs/kegg_sft_%J.err
#BSUB -L /bin/bash

## Stage 1 of 3 — Supervised Fine-Tuning (SFT) on KEGG
## -n 8           : 8 CPU slots
## mem=24000      : 24 GB/slot × 8 = ~192 GB total RAM
## num=1          : 1 exclusive GPU (A100 40GB)
## W 12:00        : 12-hour wall time (5 epochs, ~2-4h expected)
##
## Model: NT-500M + Qwen3-1.7B
## After this job: convert checkpoint with sh_convert_deepspeed_to_hf_ckpt_dna.sh
##   then submit bsub_kegg_grpo.sh

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load gcc/14.2.0
source /hpc/users/cardia04/miniconda3/etc/profile.d/conda.sh
conda activate bioreview

cd /sc/arion/work/cardia04/BioReason

export HF_HOME=/sc/arion/work/cardia04/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/sc/arion/work/cardia04/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_PROJECT=bioreason

nvidia-smi

# ---------------------------------------------------------------------------
# KEGG SFT  →  paper target (SFT only, NT-500M + Qwen3-1.7B): ~85-90% accuracy
# ---------------------------------------------------------------------------
echo "========================================================"
echo "Stage 1: KEGG SFT (NT-500M + Qwen3-1.7B)"
echo "========================================================"

stdbuf -oL -eL python train_dna_qwen.py \
    --cache_dir ${HF_HOME} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_entity andrea-cardiel-icahn \
    --text_model_name Qwen/Qwen3-1.7B \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --strategy deepspeed_stage_2 \
    --max_epochs 5 \
    --num_gpus 1 \
    --batch_size 1 \ \
    --learning_rate 5e-5 \
    --model_type dna-llm \
    --dataset_type kegg \
    --merge_val_test_set True \
    --return_answer_in_batch True \
    --checkpoint_dir /sc/arion/work/cardia04/BioReason/checkpoints/kegg_sft \
    --log_dir /sc/arion/work/cardia04/BioReason/logs

echo "========================================================"
echo "SFT complete. Checkpoint written to:"
echo "  /sc/arion/work/cardia04/BioReason/checkpoints/kegg_sft/<run>-<timestamp>/"
echo ""
echo "Next steps:"
echo "  1. Identify the timestamped checkpoint dir (ls checkpoints/kegg_sft/)"
echo "  2. Convert to HF format:"
echo "       bash sh_convert_deepspeed_to_hf_ckpt_dna.sh \\"
echo "            <ckpt_dir>/last.ckpt \\"
echo "            /sc/arion/work/cardia04/BioReason/checkpoints/kegg_sft_hf"
echo "     (Edit TEXT_MODEL_NAME=Qwen/Qwen3-1.7B and DNA_IS_EVO2=False first)"
echo "  3. Update SFT_HF_CKPT in bsub_kegg_grpo.sh and submit"
echo "========================================================"
