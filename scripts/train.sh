#!/bin/bash
# Generic training script with accelerate support
# Usage: ./train.sh [--config CONFIG_FILE] [additional args]

set -e  # Exit on error

# Default config file
CONFIG_FILE="configs/default.yaml"

# Parse arguments to find config file
ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

echo "=================================================="
echo "Mistral NER Training with Accelerate"
echo "=================================================="
echo ""
echo "Configuration: $CONFIG_FILE"
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Warning: No virtual environment detected. It's recommended to use a virtual environment."
    echo "Do you want to continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Exiting. Please activate a virtual environment and try again."
        exit 1
    fi
fi

# Check for GPU
echo "Checking GPU availability..."
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"
echo ""

# Generate accelerate config based on project config and hardware
echo "Generating accelerate configuration..."
python scripts/create_accelerate_config.py --config "$CONFIG_FILE"
echo ""

# Set environment variables for better performance
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0

# Optional: Clear cache before training (uncomment if needed)
# echo "Clearing cache directories..."
# rm -rf ~/.cache/huggingface/datasets/downloads
# rm -rf ~/.cache/huggingface/datasets/*

# Create output directories
mkdir -p logs
mkdir -p wandb_logs

# Check if accelerate config was created successfully
if [ ! -f "accelerate_config.yaml" ]; then
    echo "Error: Failed to create accelerate_config.yaml"
    exit 1
fi

# Run training with accelerate
echo "Starting training with accelerate..."
echo ""

# Build the full command
CMD="accelerate launch --config_file accelerate_config.yaml scripts/train.py --config $CONFIG_FILE ${ARGS[*]}"
echo "Command: $CMD"
echo ""

if eval "$CMD"; then
    echo ""
    echo "=================================================="
    echo "Training completed successfully!"
    echo "=================================================="
    echo ""
    
    # Extract output directory from config
    OUTPUT_DIR=$(python -c "
import yaml
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
        print(config.get('training', {}).get('final_output_dir', './mistral-ner-finetuned-final'))
except:
    print('./mistral-ner-finetuned-final')
")
    
    echo "Model saved to: $OUTPUT_DIR"
    echo ""
    echo "To run inference with the trained model:"
    echo "python scripts/inference.py --model-path $OUTPUT_DIR --text \"Your text here\""
    echo ""
    echo "To evaluate on test set:"
    echo "python scripts/evaluate.py --model-path $OUTPUT_DIR --config $CONFIG_FILE"
else
    echo ""
    echo "=================================================="
    echo "Training failed! Check the logs for errors."
    echo "=================================================="
    exit 1
fi