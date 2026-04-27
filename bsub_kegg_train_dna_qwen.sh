#!/bin/bash
#BSUB -J bioreason_kegg_train_dna_qwen
#BSUB -W 12:00
#BSUB -P acc_genome_foundation
#BSUB -q gpu
#BSUB -n 8
#BSUB -R "rusage[mem=16000] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no:gmem=40G"
#BSUB -o /sc/arion/work/cardia04/BioReason/logs/train_dna_qwen_%J.out
#BSUB -eo /sc/arion/work/cardia04/BioReason/logs/train_dna_qwen_%J.err
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


## =============================================================================
## KEGG Dataset Training
## =============================================================================

# NT-500M + Qwen3-1.7B on KEGG
stdbuf -oL -eL python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity andrea-cardiel-icahn \
    --text_model_name Qwen/Qwen3-1.7B \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --strategy deepspeed_stage_2 \
    --max_epochs 5 \
    --num_gpus 1 \
    --batch_size 1 \
    --model_type dna-llm \
    --dataset_type kegg \
    --merge_val_test_set True \
    --return_answer_in_batch True