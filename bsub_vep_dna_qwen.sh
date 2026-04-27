#!/bin/bash
#BSUB -J bioreason_vep_train_dna_qwen
#BSUB -W 12:00
#BSUB -P acc_genome_foundation
#BSUB -q gpu
#BSUB -n 8
#BSUB -R "rusage[mem=16000] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no:gmem=48G"
#BSUB -o /sc/arion/work/cardia04/BioReason/logs/vep_dna_qwen_%J.out
#BSUB -eo /sc/arion/work/cardia04/BioReason/logs/vep_dna_qwen_%J.err
#BSUB -L /bin/bash
#BSUB -u andrea.cardiel@icahn.mssm.edu
#BSUB -N

module load gcc/14.2.0
module load cuda/12.9.1

## Environment Setup
echo "CUDA_HOME: $CUDA_HOME"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "which python: $(which python)"

## Configuration Variables
# Change these to match your setup
CONDA_ENV=bioreview                     # conda environment name
CACHE_DIR=/sc/arion/work/cardia04/.cache/huggingface   # HF cache directory
WANDB_PROJECT=bioreason             # W&B project name

## Setup Environment
source /hpc/users/cardia04/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

cd /sc/arion/work/cardia04/BioReason

export HF_HOME=/sc/arion/work/cardia04/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/sc/arion/work/cardia04/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_PROJECT=bioreason

nvidia-smi

echo "======== Job started on $(hostname) at $(date) ========"

echo "======== Starting VEP Coding (NT-500M + Qwen3-4B): $(date) ========"
stdbuf -oL -eL python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity andrea-cardiel-icahn \
    --text_model_name Qwen/Qwen3-4B \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --strategy deepspeed_stage_2 \
    --max_epochs 3 \
    --num_gpus 1 \
    --batch_size 2 \
    --model_type dna-llm \
    --dataset_type variant_effect_coding \
    --return_answer_in_batch True
echo "======== VEP Coding finished (exit code: $?) at $(date) ========"