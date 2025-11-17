
#!/bin/bash

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================
NUM_SHADOWS=5
BASE_SEED=43
TOTAL_SAMPLES=50000
SAMPLES_PER_SHADOW=10000
BLOCK_SIZE=512
EPOCHS=3
BATCH_SIZE=8
LR=2e-4
LORA_R=32
LORA_ALPHA=64

DATA_DIR="wiki_json"
SHADOW_DIR="models/shadow_models"
TARGET_MODEL="models/train"  # Your target model to attack

# ============================================================================
# Step 1: Generate training data for shadow models
# ============================================================================
echo "========================================"
echo "Step 1: Generating Shadow Training Data"
echo "========================================"

# for i in $(seq 0 $((NUM_SHADOWS-1))); do
#     seed=$((BASE_SEED + i))
#     echo "Generating shadow_${i} with seed=${seed}..."
#     python prepare_data_shadow.py \
#         --seed ${seed} \
#         --shadow_id ${i} \
#         --total_samples ${TOTAL_SAMPLES} \
#         --samples_per_shadow ${SAMPLES_PER_SHADOW} \
#         --output_dir ${DATA_DIR}/train
# done


# ============================================================================
# Step 2: Train shadow models
# ============================================================================
echo "========================================"
echo "Step 2: Training Shadow Models"
echo "========================================"

for i in $(seq 0 $((NUM_SHADOWS-1))); do
    seed=$((BASE_SEED + i))
    shadow_name="shadow_${i}"
    output_dir="${SHADOW_DIR}/${shadow_name}/gpt2_3_lora32_adamw_b8_lr2"
    
    echo "Training ${shadow_name}..."
    python ft_llm/ft_llm_colab.py \
        --data_dir ${DATA_DIR}/train \
        --train_file train_shadow_${i}.json \
        --model_name gpt2 \
        --block_size ${BLOCK_SIZE} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --lora \
        --lora_r ${LORA_R} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_dropout 0.05 \
        --outdir ${output_dir} \
        --seed ${seed} \
        --gradient_accumulation_steps 1
    
    echo "✓ ${shadow_name} trained"
done



# ============================================================================
# Step 5: Run MIA Attack (Multiple Methods)
# ============================================================================
echo "========================================"
echo "Step 5: Running MIA Attack"
echo "========================================"

# Method 1: Multi-shadow with rich features
echo "Method 1: Multi-shadow with features..."
python multi_shadow_mia_complete.py \
    --target_model ${TARGET_MODEL} \
    --shadow_dir ${SHADOW_DIR} \
    --shadow_info shadow_model_info.json \
    --data_dir ${DATA_DIR}/train \
    --data_file train_shadow_0.json \
    --sample_indices_file sample_indices.npy \
    --label_file label.npy \
    --method features \
    --output predictions_features.npy