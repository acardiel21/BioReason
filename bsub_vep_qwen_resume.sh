#!/bin/bash
#BSUB -J bioreason_vep_qwen_resume
#BSUB -P acc_genome_foundation
#BSUB -q gpu
#BSUB -n 4
#BSUB -R "rusage[mem=48000] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no:gmem=40G"
#BSUB -W 24:00
#BSUB -o /sc/arion/work/cardia04/BioReason/logs/vep_qwen_resume_%J.out
#BSUB -eo /sc/arion/work/cardia04/BioReason/logs/vep_qwen_resume_%J.err
#BSUB -L /bin/bash

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load gcc/14.2.0
module load cuda/13.0.0
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
# Run 1: VEP Coding — resume from epoch 1 checkpoint, complete to epoch 3
# ---------------------------------------------------------------------------
echo "========================================================"
echo "Resuming VEP Coding from checkpoint (epochs 2-3 + test)"
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
    --dataset_type variant_effect_coding \
    --return_answer_in_batch True \
    --ckpt_path "checkpoints/bioreason-variant_effect_coding-Qwen3-4B-20260425-162815/bioreason-variant_effect_coding-Qwen3-4B-epoch=00-val_loss_epoch=nan.ckpt"

# ---------------------------------------------------------------------------
# Run 2: VEP Non-SNV — train from scratch (1 epoch) + test
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
    --dataset_type variant_effect_non_snv \
    --return_answer_in_batch True

echo "========================================================"
echo "Both VEP Qwen runs complete."
echo "========================================================"
