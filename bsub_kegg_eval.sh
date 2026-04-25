#!/bin/bash
#BSUB -J bioreason_kegg_eval
#BSUB -P acc_genome_foundation
#BSUB -q gpu
#BSUB -n 6
#BSUB -R "rusage[mem=21000]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -W 4:00
#BSUB -o /sc/arion/work/cardia04/BioReason/logs/kegg_eval_%J.out
#BSUB -eo /sc/arion/work/cardia04/BioReason/logs/kegg_eval_%J.err
#BSUB -L /bin/bash

## Stage 3 of 3 — KEGG Evaluation with vLLM
## -n 6           : 6 CPU slots
## mem=21000      : 21 GB/slot × 6 = ~126 GB total RAM
## num=1          : 1 exclusive GPU (A100 40GB)
## W 4:00         : 4-hour wall time (290 examples, ~1-2h expected)
##
## Model: NT-500M + Qwen3-1.7B (evaluates the HF-converted GRPO checkpoint)
## PREREQ: Run bsub_kegg_grpo.sh and convert with sh_convert_grpo_to_hf_ckpt.sh
## Can also be used to evaluate the SFT-only checkpoint (for comparison)

# ---------------------------------------------------------------------------
# *** SET THIS PATH BEFORE SUBMITTING ***
# ---------------------------------------------------------------------------
GRPO_HF_CKPT=/sc/arion/work/cardia04/BioReason/checkpoints/kegg_grpo_hf
EVAL_OUTPUT=/sc/arion/work/cardia04/BioReason/eval_results/kegg_nt_qwen17b_grpo

# To evaluate SFT-only checkpoint instead, set:
# GRPO_HF_CKPT=/sc/arion/work/cardia04/BioReason/checkpoints/kegg_sft_hf
# EVAL_OUTPUT=/sc/arion/work/cardia04/BioReason/eval_results/kegg_nt_qwen17b_sft

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

nvidia-smi

echo "========================================================"
echo "Stage 3: KEGG Evaluation (NT-500M + Qwen3-1.7B)"
echo "Checkpoint: ${GRPO_HF_CKPT}"
echo "Output:     ${EVAL_OUTPUT}"
echo "Paper target (GRPO, NT-500M + Qwen3-4B): 98.28% accuracy, 93.05 F1"
echo "========================================================"

stdbuf -oL -eL python eval_kegg_dna_vllm.py \
    --ckpt_dir ${GRPO_HF_CKPT} \
    --cache_dir ${HF_HOME} \
    --text_model_name Qwen/Qwen3-1.7B \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --output_dir ${EVAL_OUTPUT} \
    --max_length_dna 2048 \
    --max_length_text 1024 \
    --truncate_dna_per_side 1024 \
    --temperature 0 \
    --top_p 0.95 \
    --max_new_tokens 800 \
    --gpu_memory_utilization 0.5 \
    --dna_is_evo2 False

echo "========================================================"
echo "Evaluation complete. Results saved to: ${EVAL_OUTPUT}"
echo "Check *_kegg_eval_metrics_*.json for accuracy and F1."
echo "Check *_kegg_eval_results_*.csv for per-example reasoning traces."
echo "========================================================"
