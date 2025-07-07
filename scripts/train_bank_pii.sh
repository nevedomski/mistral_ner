#!/bin/bash
# Training script for Bank PII Detection Model
# Uses multi-dataset training with CoNLL-2003, OntoNotes, Gretel PII, and AI4Privacy

set -e  # Exit on error

echo "=================================================="
echo "Starting Bank PII Detection Model Training"
echo "=================================================="
echo ""
echo "Configuration: configs/bank_pii.yaml"
echo "Datasets: CoNLL-2003, OntoNotes, Gretel PII, AI4Privacy"
echo "Output: ./mistral-ner-bank-pii-final"
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

# Set environment variables for better performance
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0

# Optional: Clear cache before training
echo "Clearing cache directories..."
rm -rf ~/.cache/huggingface/datasets/downloads
rm -rf ~/.cache/huggingface/datasets/conll2003
rm -rf ~/.cache/huggingface/datasets/tner
rm -rf ~/.cache/huggingface/datasets/gretelai
rm -rf ~/.cache/huggingface/datasets/ai4privacy

# Create output directories
mkdir -p logs/bank_pii
mkdir -p wandb_logs

# Run training with the bank PII configuration
echo "Starting training..."
echo ""

if python scripts/train.py \
    --config configs/bank_pii.yaml \
    "$@"; then  # Pass any additional arguments
    echo ""
    echo "=================================================="
    echo "Training completed successfully!"
    echo "=================================================="
    echo ""
    echo "Model saved to: ./mistral-ner-bank-pii-final"
    echo ""
    echo "To run inference with the trained model:"
    echo "python scripts/inference.py --model-path ./mistral-ner-bank-pii-final --text \"Your text here\""
    echo ""
    echo "To evaluate on test set:"
    echo "python scripts/evaluate.py --model-path ./mistral-ner-bank-pii-final --config configs/bank_pii.yaml"
else
    echo ""
    echo "=================================================="
    echo "Training failed! Check the logs for errors."
    echo "=================================================="
    exit 1
fi