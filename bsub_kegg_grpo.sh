#!/bin/bash
#BSUB -J bioreason_kegg_grpo
#BSUB -P acc_genome_foundation
#BSUB -q gpu
#BSUB -n 16
#BSUB -R "rusage[mem=20000]"
#BSUB -gpu "num=2:mode=exclusive_process:mps=no"
#BSUB -W 24:00
#BSUB -o /sc/arion/work/cardia04/BioReason/logs/kegg_grpo_%J.out
#BSUB -eo /sc/arion/work/cardia04/BioReason/logs/kegg_grpo_%J.err
#BSUB -L /bin/bash

## Stage 2 of 3 — GRPO Reinforcement Learning on KEGG
## -n 16          : 16 CPU slots (8 per GPU)
## mem=20000      : 20 GB/slot × 16 = ~320 GB total RAM
## num=2          : 2 exclusive GPUs (A100 40GB each)
## W 24:00        : 24-hour wall time (1000 steps, ~12-18h expected)
##
## Model: NT-500M + Qwen3-1.7B (starts from HF-converted SFT checkpoint)
## GPU split: rank 0 = trainer, rank 1 = vLLM colocate (30% GPU mem each)
## PREREQ: Run bsub_kegg_sft.sh and convert checkpoint with
##         sh_convert_deepspeed_to_hf_ckpt_dna.sh before submitting this job

# ---------------------------------------------------------------------------
# *** SET THIS PATH BEFORE SUBMITTING ***
# ---------------------------------------------------------------------------
SFT_HF_CKPT=/sc/arion/work/cardia04/BioReason/checkpoints/kegg_sft_hf
GRPO_OUTPUT=/sc/arion/work/cardia04/BioReason/checkpoints/kegg_grpo_nt_qwen17b

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

# NCCL tuning for single-node multi-GPU
export OMP_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

MASTER_PORT=$((12000 + LSB_JOBID % 20000))

nvidia-smi

echo "========================================================"
echo "Stage 2: KEGG GRPO (NT-500M + Qwen3-1.7B)"
echo "SFT checkpoint: ${SFT_HF_CKPT}"
echo "Output:         ${GRPO_OUTPUT}"
echo "Paper target (GRPO, NT-500M + Qwen3-1.7B): ~95%+ accuracy"
echo "========================================================"

torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --master_addr=127.0.0.1 \
    --master_port=${MASTER_PORT} \
    train_grpo.py \
    --text_model_name Qwen/Qwen3-1.7B \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --cache_dir ${HF_HOME} \
    --sft_checkpoint "${SFT_HF_CKPT}" \
    --peft_ckpt False \
    --dna_is_evo2 False \
    --truncate_dna_per_side 0 \
    --deepspeed grpo_trainer_lora_model/ds_config_stage2.json \
    --lora_r 16 --lora_alpha 32 --lora_dropout 0 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --max_steps 1000 \
    --max_completion_length 800 \
    --num_generations 8 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --beta 0.0 \
    --run_name kegg-grpo-nt500m-qwen17b \
    --learning_rate 1e-5 \
    --logging_steps 1 \
    --temperature 1 \
    --top_p 0.95 \
    --top_k 20 \
    --output_dir "${GRPO_OUTPUT}" \
    --save_strategy steps --save_steps 100 --save_total_limit 2 \
    --lr_scheduler_type cosine --warmup_ratio 0.03 \
    --log_completions True \
    --use_vllm True \
    --vllm_mode colocate \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_max_model_len 3000 \
    --vllm_ckpt "${SFT_HF_CKPT}" \
    --bf16 True \
    --resume_from_checkpoint True

echo "========================================================"
echo "GRPO complete. Checkpoint written to: ${GRPO_OUTPUT}"
echo ""
echo "Next steps:"
echo "  1. Identify the best checkpoint dir (ls ${GRPO_OUTPUT}/)"
echo "  2. Convert to HF format:"
echo "       bash sh_convert_grpo_to_hf_ckpt.sh \\"
echo "            ${GRPO_OUTPUT}/checkpoint-<step> \\"
echo "            /sc/arion/work/cardia04/BioReason/checkpoints/kegg_grpo_hf"
echo "     (Edit TEXT_MODEL_NAME=Qwen/Qwen3-1.7B and DNA_IS_EVO2=False first)"
echo "  3. Update GRPO_HF_CKPT in bsub_kegg_eval.sh and submit"
echo "========================================================"
