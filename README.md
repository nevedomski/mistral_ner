# mistral-ner

Fine-tune the Mistral-7B model for Named Entity Recognition (NER) on the CoNLL-2003 dataset using Hugging Face Transformers.

## Features

- Loads and preprocesses the CoNLL-2003 NER dataset
- Fine-tunes a Mistral-7B model for token classification
- Evaluates using seqeval metrics (precision, recall, F1, accuracy)
- Uses Hugging Face Trainer API

## Installation

```bash
uv pip install .
```

## Requirements

- Python >= 3.11
- See dependencies in `pyproject.toml`

## Usage

To fine-tune the model on CoNLL-2003:

```bash
python finetune_conll2023.py
```

This will train the model and save outputs to `./mistral-ner-finetuned`.

## Multi-GPU Training

To train on multiple GPUs, use the Hugging Face Accelerate launcher or PyTorch's `torchrun`:

```bash
uv pip install accelerate
accelerate launch finetune_conll2023.py
```
or
```bash
torchrun --nproc_per_node=NUM_GPUS finetune_conll2023.py
```

The script will automatically use all available GPUs.

## Configuration

- The model checkpoint is set to `mistralai/Mistral-7B-v0.3` by default. Change `model_checkpoint` in `finetune_conll2023.py` to use a different checkpoint.
- Training arguments (batch size, epochs, etc.) can be modified in the script.

## Output

- Trained model and checkpoints in `./mistral-ner-finetuned`
- Training logs in `./logs`
- Evaluation metrics printed after training

## License

MIT License
