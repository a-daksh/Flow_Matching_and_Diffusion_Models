#!/bin/bash

# Ablation Study Script for Flow Matching
# This script trains models with different data fractions and runs inference with
# different guidance scales and number of steps.
#
# Usage:
#   ./run_ablation.sh                    # Run full ablation (training + inference)
#   ./run_ablation.sh train              # Only train models
#   ./run_ablation.sh inference          # Only run inference (assumes models are trained)

set -e  # Exit on error

# Configuration
METHOD="flow"
BASE_DIR="checkpoints"
OUTPUT_DIR="outputs"
SEED=42
NUM_EPOCHS=5000
BATCH_SIZE=250
LR=1e-3
ETA=0.1

DATA_FRACTIONS=(0.1 0.25 0.5 1.0)

GUIDANCE_SCALES=(0.5 1.0 2.0 3.0 5.0)

INFERENCE_STEPS=(10 25 50 100 200)

SAMPLES_PER_CLASS=10

MODE="${1:-all}"  # Default to "all" if no argument provided

echo "=========================================="
echo "Flow Matching Ablation Study"
echo "=========================================="
echo "Mode: ${MODE}"
echo "Data fractions: ${DATA_FRACTIONS[@]}"
echo "Guidance scales: ${GUIDANCE_SCALES[@]}"
echo "Inference steps: ${INFERENCE_STEPS[@]}"
echo "=========================================="
echo ""
# Parse command line argument

# Create output directory
mkdir -p ${OUTPUT_DIR}

# ==========================================
# PHASE 1: TRAINING
# ==========================================
if [ "${MODE}" == "all" ] || [ "${MODE}" == "train" ]; then
    echo "PHASE 1: Training models with different data fractions..."
    echo ""
    
    for frac in "${DATA_FRACTIONS[@]}"; do
        # Create descriptive postfix
        postfix="data_frac_${frac}"
        
        echo "Training model with data fraction: ${frac} (postfix: ${postfix})"
        
        python main.py train \
            --method ${METHOD} \
            --seed ${SEED} \
            --num_epochs ${NUM_EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --lr ${LR} \
            --train_fraction ${frac} \
            --eta ${ETA} \
            --checkpoint_base_dir ${BASE_DIR} \
            --postfix ${postfix} \
            --checkpoint_every 100
        
        echo "✓ Completed training for data fraction ${frac}"
        echo ""
    done
    
    echo "=========================================="
    echo "All training completed!"
    echo "=========================================="
    echo ""
fi

# ==========================================
# PHASE 2: INFERENCE
# ==========================================
if [ "${MODE}" == "all" ] || [ "${MODE}" == "inference" ]; then
    echo "PHASE 2: Running inference with different configurations..."
    echo ""
    
    for frac in "${DATA_FRACTIONS[@]}"; do
        postfix="data_frac_${frac}"
        checkpoint_path="checkpoint_${METHOD}_${postfix}"
        
        # Check if checkpoint exists
        checkpoint_dir="${BASE_DIR}/${checkpoint_path}"
        if [ ! -f "${checkpoint_dir}/model.pt" ]; then
            echo "⚠ Warning: Checkpoint not found: ${checkpoint_path}"
            echo "  Skipping inference for this model."
            echo ""
            continue
        fi
        
        echo "Processing model: ${checkpoint_path}"
        
        # Create output subdirectory for this data fraction
        output_subdir="${OUTPUT_DIR}/${postfix}"
        mkdir -p ${output_subdir}
        
        # Test different number of inference steps
        for steps in "${INFERENCE_STEPS[@]}"; do
            echo "  Running inference with ${steps} steps..."
            
            # Run inference with all guidance scales
            # Note: plt.show() will be called but won't block in non-interactive mode
            python main.py inference \
                --checkpoint_path ${checkpoint_path} \
                --checkpoint_base_dir ${BASE_DIR} \
                --num_timesteps ${steps} \
                --guidance_scales ${GUIDANCE_SCALES[@]} \
                --samples_per_class ${SAMPLES_PER_CLASS} \
                --output_dir ${output_subdir} 2>&1 | grep -v "UserWarning\|matplotlib" || true
            
            # The output will be saved as: inference_${METHOD}_${postfix}.png
            # Rename to include step count for better organization
            original_file="${output_subdir}/inference_${METHOD}_${postfix}.png"
            new_file="${output_subdir}/inference_steps_${steps}.png"
            
            if [ -f "${original_file}" ]; then
                mv "${original_file}" "${new_file}"
                echo "    ✓ Saved to ${new_file}"
            else
                echo "    ⚠ Warning: Expected output file not found: ${original_file}"
            fi
        done
        
        echo "  ✓ Completed inference for ${checkpoint_path}"
        echo ""
    done
    
    echo "=========================================="
    echo "All inference completed!"
    echo "=========================================="
    echo ""
fi

