# Changelog

All notable changes to Mistral NER will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive MkDocs documentation with Material theme
- Support for 9 datasets including traditional NER and PII detection
- Multi-dataset training with three mixing strategies
- Advanced loss functions for handling class imbalance
- Hyperparameter optimization with Ray Tune and Optuna
- 8-bit and 4-bit quantization support
- WandB integration with offline mode support
- Validation script with comprehensive reporting
- GitHub Actions CI/CD pipeline

### Changed
- Default quantization from 8-bit to 4-bit for better memory efficiency
- Enhanced LoRA configuration with more target modules
- Improved error handling and logging throughout

### Fixed
- 8-bit quantization now works correctly when 4-bit is disabled
- Memory leak issues during long training runs
- Label alignment issues with subword tokenization

## [0.2.0] - 2024-01-15

### Added
- Initial public release
- Basic Mistral-7B fine-tuning for NER
- CoNLL-2003 dataset support
- LoRA and 8-bit quantization
- Basic training and inference scripts

### Security
- No known security issues

---

For detailed changes, see the [Git commit history](https://github.com/nevedomski/mistral_ner/commits/main).