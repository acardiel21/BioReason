#!/bin/bash
#BSUB -J bioreason_vep_qwen4b
#BSUB -P acc_genome_foundation
#BSUB -q gpu
#BSUB -n 8
#BSUB -R "rusage[mem=24000]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -W 24:00
#BSUB -o /sc/arion/work/cardia04/BioReason/logs/vep_qwen_%J.out
#BSUB -eo /sc/arion/work/cardia04/BioReason/logs/vep_qwen_%J.err
#BSUB -L /bin/bash

## -n 8           : 8 CPU slots (more workers needed for text tokenization)
## mem=24000      : 24 GB/slot × 8 = ~192 GB total RAM
##                  (Qwen3-4B ~8GB + NT-500M ~2GB + activations + LoRA)
## num=1          : 1 exclusive GPU (A100 40GB)
## W 24:00        : 24-hour wall time (two runs, each ~8-10h with batch_size=1)

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
# Run 1: VEP Coding  (DNA + Qwen3-1.7B reasoning)
# ---------------------------------------------------------------------------
echo "========================================================"
echo "Starting VEP Coding (NT-500M + Qwen3-4B)"
echo "========================================================"

stdbuf -oL -eL python train_dna_qwen.py \
    --cache_dir ${HF_HOME} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_entity andrea-cardiel-icahn \
    --text_model_name Qwen/Qwen3-4B \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --strategy deepspeed_stage_2 \
    --max_epochs 3 \
    --num_gpus 1 \
    --batch_size 2 \
    --model_type dna-llm \
    --dataset_type variant_effect_coding

# ---------------------------------------------------------------------------
# Run 2: VEP Non-SNV  (DNA + Qwen3-4B reasoning)
# ---------------------------------------------------------------------------
echo "========================================================"
echo "Starting VEP Non-SNV (NT-500M + Qwen3-4B)"
echo "========================================================"

stdbuf -oL -eL python train_dna_qwen.py \
    --cache_dir ${HF_HOME} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_entity andrea-cardiel-icahn \
    --text_model_name Qwen/Qwen3-4B \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --strategy deepspeed_stage_2 \
    --max_epochs 1 \
    --num_gpus 1 \
    --batch_size 2 \
    --model_type dna-llm \
    --dataset_type variant_effect_non_snv

echo "========================================================"
echo "Both VEP Qwen runs complete."
echo "========================================================"
