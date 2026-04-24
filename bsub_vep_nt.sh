#!/bin/bash
#BSUB -J bioreason_vep_nt
#BSUB -q gpu
#BSUB -n 6
#BSUB -R "rusage[mem=21000]"
#BSUB -R "select[gpu_model0==NVIDIAA100_SXM4]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -P acc_genome_foundation
#BSUB -W 12:00
#BSUB -o /sc/arion/work/cardia04/BioReason/logs/vep_nt_%J.out
#BSUB -e /sc/arion/work/cardia04/BioReason/logs/vep_nt_%J.err

## -n 6           : 6 CPU slots
## mem=21000      : 21 GB/slot × 6 = ~126 GB total RAM
## ngpus_excl_p=1 : 1 exclusive GPU
## W 12:00        : 12-hour wall time (two sequential 3-epoch runs)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load gcc/14.2.0
source /hpc/users/cardia04/miniconda3/etc/profile.d/conda.sh
conda activate bioreview

cd /sc/arion/work/cardia04/BioReason

export HF_HOME=/sc/arion/work/cardia04/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=bioreason

nvidia-smi

# ---------------------------------------------------------------------------
# Run 1: VEP Coding  →  paper target: Accuracy 60.91, F1 45.20
# ---------------------------------------------------------------------------
echo "========================================================"
echo "Starting VEP Coding (NT-500M)"
echo "========================================================"

stdbuf -oL -eL python train_dna_only.py \
    --cache_dir ${HF_HOME} \
    --wandb_project ${WANDB_PROJECT} \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --strategy ddp \
    --max_epochs 3 \
    --num_gpus 1 \
    --batch_size 2 \
    --max_length_dna 2048 \
    --truncate_dna_per_side 1024 \
    --train_just_classifier True \
    --learning_rate 3e-4 \
    --dataset_type variant_effect_coding

# ---------------------------------------------------------------------------
# Run 2: VEP Non-SNV  →  paper target: Accuracy 67.93, F1 65.97
# ---------------------------------------------------------------------------
echo "========================================================"
echo "Starting VEP Non-SNV (NT-500M)"
echo "========================================================"

stdbuf -oL -eL python train_dna_only.py \
    --cache_dir ${HF_HOME} \
    --wandb_project ${WANDB_PROJECT} \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --strategy ddp \
    --max_epochs 3 \
    --num_gpus 1 \
    --batch_size 2 \
    --max_length_dna 2048 \
    --truncate_dna_per_side 1024 \
    --train_just_classifier True \
    --learning_rate 3e-4 \
    --dataset_type variant_effect_non_snv

echo "========================================================"
echo "Both VEP runs complete."
echo "========================================================"
