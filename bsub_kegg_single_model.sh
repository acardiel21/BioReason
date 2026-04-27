#!/bin/bash
#BSUB -J bioreason_kegg_train_single_models # name of job
#BSUB -W 12:00 # time limit
#BSUB -P acc_genome_foundation # project 
#BSUB -q gpu # partition 
#BSUB -n 8 # number of cores
#BSUB -R "rusage[mem=16000] span[hosts=1]" # memory limit
#BSUB -gpu "num=1:mode=exclusive_process:mps=no:gmem=40G"
#BSUB -o /sc/arion/work/cardia04/BioReason/logs/kegg_train_single_%J.out
#BSUB -eo /sc/arion/work/cardia04/BioReason/logs/kegg_train_single_%J.err
#BSUB -L /bin/bash

module load gcc/14.2.0
module load cuda/13.0.0

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

export HF_HOME=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

nvidia-smi                             # Check GPU status

# NT-500M on KEGG (DNA-only)
stdbuf -oL -eL python train_dna_only.py \
    --cache_dir $CACHE_DIR \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity andrea-cardiel-icahn \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --strategy ddp \
    --max_epochs 5 \
    --num_gpus 1 \
    --batch_size 1 \
    --max_length_dna 2048 \
    --truncate_dna_per_side 1024 \
    --train_just_classifier True \
    --learning_rate 3e-4 \
    --dataset_type kegg \
    --merge_val_test_set True


# Qwen3-4B on KEGG (LLM-only)
stdbuf -oL -eL python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity andrea-cardiel-icahn \
    --text_model_name Qwen/Qwen3-4B \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --strategy deepspeed_stage_2 \
    --max_epochs 5 \
    --num_gpus 1 \
    --batch_size 1 \
    --model_type llm \
    --dataset_type kegg \
    --max_length_dna 4 \
    --max_length_text 8192 \
    --truncate_dna_per_side 1024 \
    --merge_val_test_set True \
    --return_answer_in_batch True



