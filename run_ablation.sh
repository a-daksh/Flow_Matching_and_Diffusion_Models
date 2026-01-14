#!/bin/bash
set -e

# Configuration
BASE_DIR="checkpoints"
OUTPUT_DIR="outputs"
NUM_EPOCHS=5000
BATCH_SIZE=250
LR=1e-3
ETA=0.1

DATA_FRACTIONS=(0.1 0.25 0.5 0.75 1.0)
GUIDANCE_SCALES_STUDY_B=(0.5 1.0 2.0 3.0 5.0)
INFERENCE_STEPS_STUDY_B=(10 25 50 100 200)
SIGMA_VALUES=(0 0.1 0.2 0.5 1.0)
SAMPLES_PER_CLASS=10

MODE="${1:-all}"

echo "=========================================="
echo "Modepleple: ${MODE}"
echo "=========================================="

mkdir -p ${OUTPUT_DIR}

# ==========================================
# TRAINING: 5 models with different data fractions
# ==========================================
if [ "${MODE}" == "all" ] || [ "${MODE}" == "train" ]; then
    echo "Training models..."
    
    for frac in "${DATA_FRACTIONS[@]}"; do
        postfix="frac_${frac}"
        
        python main.py train \
            --num_epochs ${NUM_EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --lr ${LR} \
            --train_fraction ${frac} \
            --eta ${ETA} \
            --checkpoint_base_dir ${BASE_DIR} \
            --postfix ${postfix} \
            --checkpoint_every 100
        
        echo "✓ Trained: ${postfix}"
    done
    
    echo "Training complete"
    echo ""
fi

# ==========================================
# STUDY A: Data Fraction Effect
# Baseline inference on all 5 models (flow only, w=3.0, steps=100)
# ==========================================
if [ "${MODE}" == "all" ] || [ "${MODE}" == "study_a" ]; then
    echo "Study A: Data Fraction Effect"
    
    output_subdir="${OUTPUT_DIR}/study_a_data_fraction"
    mkdir -p ${output_subdir}
    
    for frac in "${DATA_FRACTIONS[@]}"; do
        postfix="frac_${frac}"
        checkpoint_path="checkpoint_${postfix}"
        
        if [ ! -f "${BASE_DIR}/${checkpoint_path}/model.pt" ]; then
            echo "⚠ Checkpoint not found: ${checkpoint_path}"
            continue
        fi
        
        python main.py inference \
            --checkpoint_path ${checkpoint_path} \
            --checkpoint_base_dir ${BASE_DIR} \
            --num_timesteps 100 \
            --guidance_scales 3.0 \
            --samples_per_class ${SAMPLES_PER_CLASS} \
            --output_dir ${output_subdir}
        
        echo "✓ Study A: frac=${frac}"
    done
    
    echo "Study A complete"
    echo ""
fi

# ==========================================
# STUDY B: Flow Parameters (Guidance + Steps)
# Deep dive on frac=1.0 model
# ==========================================
if [ "${MODE}" == "all" ] || [ "${MODE}" == "study_b" ]; then
    echo "Study B: Flow Parameters (Guidance + Steps)"
    
    postfix="frac_1.0"
    checkpoint_path="checkpoint_${postfix}"
    output_subdir="${OUTPUT_DIR}/study_b_flow_params"
    mkdir -p ${output_subdir}
    
    if [ ! -f "${BASE_DIR}/${checkpoint_path}/model.pt" ]; then
        echo "⚠ Checkpoint not found: ${checkpoint_path}"
    else
        for w in "${GUIDANCE_SCALES_STUDY_B[@]}"; do
            for steps in "${INFERENCE_STEPS_STUDY_B[@]}"; do
                python main.py inference \
                    --checkpoint_path ${checkpoint_path} \
                    --checkpoint_base_dir ${BASE_DIR} \
                    --num_timesteps ${steps} \
                    --guidance_scales ${w} \
                    --samples_per_class ${SAMPLES_PER_CLASS} \
                    --output_dir ${output_subdir}
                
                original_file="${output_subdir}/inference_ode_${postfix}.png"
                new_file="${output_subdir}/inference_ode_w${w}_steps${steps}_${postfix}.png"
                
                if [ -f "${original_file}" ]; then
                    mv "${original_file}" "${new_file}"
                fi
                
                echo "✓ Study B: w=${w}, steps=${steps}"
            done
        done
    fi
    
    echo "Study B complete"
    echo ""
fi

# ==========================================
# STUDY C: Diffusion Sigma Effect
# Deep dive on frac=1.0 model (guidance=3.0, steps=100, vary sigma)
# ==========================================
if [ "${MODE}" == "all" ] || [ "${MODE}" == "study_c" ]; then
    echo "Study C: Diffusion Sigma Effect"
    
    postfix="frac_1.0"
    checkpoint_path="checkpoint_${postfix}"
    output_subdir="${OUTPUT_DIR}/study_c_diffusion_sigma"
    mkdir -p ${output_subdir}
    
    if [ ! -f "${BASE_DIR}/${checkpoint_path}/model.pt" ]; then
        echo "⚠ Checkpoint not found: ${checkpoint_path}"
    else
        for sigma in "${SIGMA_VALUES[@]}"; do
            if [ "${sigma}" == "0" ]; then
                python main.py inference \
                    --checkpoint_path ${checkpoint_path} \
                    --checkpoint_base_dir ${BASE_DIR} \
                    --num_timesteps 100 \
                    --guidance_scales 3.0 \
                    --samples_per_class ${SAMPLES_PER_CLASS} \
                    --output_dir ${output_subdir}
                
                original_file="${output_subdir}/inference_ode_${postfix}.png"
                new_file="${output_subdir}/inference_sigma${sigma}_${postfix}.png"
            else
                python main.py inference \
                    --checkpoint_path ${checkpoint_path} \
                    --checkpoint_base_dir ${BASE_DIR} \
                    --num_timesteps 100 \
                    --guidance_scales 3.0 \
                    --samples_per_class ${SAMPLES_PER_CLASS} \
                    --stochastic \
                    --sigma ${sigma} \
                    --output_dir ${output_subdir}
                
                original_file="${output_subdir}/inference_sde_sigma${sigma}_${postfix}.png"
                new_file="${output_subdir}/inference_sigma${sigma}_${postfix}.png"
            fi
            
            if [ -f "${original_file}" ]; then
                mv "${original_file}" "${new_file}"
            fi
            
            echo "✓ Study C: sigma=${sigma}"
        done
    fi
    
    echo "Study C complete"
    echo ""
fi

echo "=========================================="
echo "All studies complete"
echo "=========================================="
