# AGENTS.md - Mistral NER v2 Project

## Project Overview

**mistral_ner_v2** is a comprehensive Named Entity Recognition (NER) fine-tuning framework for the Mistral-7B model. The project specializes in multi-dataset training with flexible dataset loading, tag scheme conversion, and unified entity type mapping. It uses Hugging Face Transformers with QLoRA for efficient fine-tuning.

## Current Status

- **Refactoring Progress**: ~85% complete
- **Focus Area**: `src/dataset_utils/` folder - core dataset infrastructure
- **Architecture**: Transitioning from legacy pattern to Pydantic dataclass-based loaders
- **Last Updated**: 2025-08-05 (Comprehensive review conducted, context management initialized)

## Critical Security Vulnerabilities ⚠️

### Immediate Action Required

1. **Code Injection Vulnerabilities**
   - **ai4privacy.py (Line 196)**: `ast.literal_eval()` on untrusted dataset strings
   - **mendeley_pii.py (Line 214)**: `ast.literal_eval()` on untrusted dataset strings
   - **Risk**: Arbitrary code execution if malicious strings in datasets
   - **Priority**: HIGH - Replace with safer JSON parsing

2. **Network Security Issues**
   - **ontonotes.py (Line 32)**: Unvalidated HTTP requests to external URLs
   - **Risk**: SSRF attacks, DNS poisoning, certificate bypass
   - **Priority**: MEDIUM - Add URL validation and certificate verification

3. **Input Validation Gaps**
   - **base.py (Line 155-162)**: No validation of id2label function parameters
   - **Risk**: Potential arbitrary code execution through malicious callable
   - **Priority**: MEDIUM - Add strict input validation

## Memory Management Issues

1. **Out-of-Memory Risks**
   - Large datasets loaded entirely into memory (Mendeley: 200k, AI4Privacy: 65k examples)
   - No streaming support implemented
   - **Risk**: OOM crashes in production environments
   - **DOCDEV-TODO**: Implement IterableDataset support and memory monitoring

2. **Resource Leaks**
   - Dataset objects not properly cleaned up after use
   - Multiple datasets can accumulate in memory during multi-dataset training

## Incomplete Components (High Risk for Runtime Errors)

1. **_mixers.py**: Partial implementation of dataset mixing strategies
2. **_multi_dataset.py**: Uses legacy architecture, incompatible with new loaders
3. **_samplers.py**: Incomplete multi-dataset sampling logic
4. **registry.py**: Outdated pattern, needs refactoring for dataclass loaders

## Architecture Overview

### Dataset Loading Pipeline
```
Dataset Source (HF Hub) → BaseDataset → Tag Conversion → Label Mapping → Validation
                     ↓
              Preprocess() → ClassLabel Format → Unified Schema → Training Ready
```

### Key Components

1. **BaseDataset** (`src/dataset_utils/base.py`)
   - Abstract base class for all dataset loaders
   - Handles tag scheme conversion, label mapping, validation
   - **Security Issues**: Missing input validation for id2label

2. **Dataset Loaders** (`src/dataset_utils/loaders/`)
   - 9 dataset loaders covering standard NER and PII detection
   - **Security Issues**: ast.literal_eval vulnerabilities in PII loaders

3. **TagSchemeConverter** (`src/dataset_utils/converter.py`)
   - Converts between NER tagging schemes (IO, IOB, BIO, BIOES, BILOU)
   - **Risk**: Potential infinite loops in chained conversions

4. **Configuration System** (`src/config*.py`)
   - Pydantic-based configuration with Hydra integration
   - **Issue**: Tight coupling through hardcoded imports

## Tag Processing Pipeline (3-Level System)

1. **Original Tags** (`ner_tags`): Dataset's native format
2. **Converted Tags** (`single_set_tags`): Converted to consistent scheme (BIO default)
3. **Unified Tags** (`single_set_unified_tags`): Mapped to unified entity schema

### Unified Entity Schema
- **Standard**: PER, ORG, LOC, MISC, GPE, FAC
- **Banking/PII**: CARD, BANK, SSN, EMAIL, PHONE, ADDR
- **Temporal**: DATE, TIME
- **Numeric**: MONEY, CARDINAL, ORDINAL

## Development Guidelines

### Security-First Approach
1. **Never use `ast.literal_eval()` on untrusted data**
2. **Validate all network requests** (URLs, certificates, timeouts)
3. **Implement strict input validation** for all external data
4. **Use streaming for large datasets** to prevent OOM errors

### Code Quality Standards
- **Type Safety**: Full mypy compliance with strict mode
- **Error Handling**: Comprehensive exception handling with proper logging
- **Memory Management**: Implement cleanup and streaming patterns
- **Testing**: 90%+ coverage for all critical components

### Adding New Datasets
1. Create loader in `src/dataset_utils/loaders/`
2. Inherit from `BaseDataset` and implement `preprocess()`
3. Define `label_mappings` for unified schema
4. **Security Check**: Avoid `ast.literal_eval()` and network requests
5. Test with `test_all_loaders.py`

## Performance Considerations

### Memory Usage by Dataset Size
- **Small** (<50k): CoNLL, WNUT, WikiNER - Safe for memory loading
- **Medium** (50k-100k): AI4Privacy - Monitor memory usage
- **Large** (>100k): Mendeley, OntoNotes - Requires streaming or sampling

### Optimization Strategies
1. **Early Language Filtering**: Reduce memory footprint for multilingual datasets
2. **Sampling Limits**: Use `max_samples_per_dataset` for development
3. **Streaming**: Implement IterableDataset for production environments

## Testing and Validation

### Current Test Coverage
- **test_all_loaders.py**: Basic loading and validation tests  
- **Coverage**: ~70% for dataset utilities
- **Missing**: Security testing, memory testing, edge cases

### Validation Levels
1. **Dataset Schema**: Converted tags match expected format
2. **Unified Schema**: Mapped tags follow unified entity types
3. **Memory Safety**: OOM prevention and cleanup validation

## Known Technical Debt

1. **IterableDataset Support**: NotImplementedError in base class
2. **Language Filtering**: Inconsistent implementation across loaders
3. **Token Alignment**: Simple whitespace tokenization in PII datasets
4. **Caching**: No persistent cache for network-fetched data (OntoNotes)
5. **Error Recovery**: Limited fallback mechanisms for failed operations

## Next Steps (Priority Order)

### 1. Security Fixes (Immediate)
- [ ] Replace `ast.literal_eval()` with safe JSON parsing in PII loaders
- [ ] Add URL validation and certificate verification for network requests
- [ ] Implement input validation for all external data processing

### 2. Memory Management (High Priority)
- [ ] Implement IterableDataset support in BaseDataset
- [ ] Add memory monitoring and automatic cleanup
- [ ] Create streaming variants for large datasets

### 3. Complete Refactoring (Medium Priority)
- [ ] Refactor incomplete components (_mixers.py, _multi_dataset.py, etc.)
- [ ] Update registry pattern for new dataclass loaders
- [ ] Standardize language filtering across all loaders

### 4. Infrastructure Improvements
- [ ] Add comprehensive security testing
- [ ] Implement persistent caching for network data
- [ ] Create plugin-based dataset discovery system
- [ ] Add performance benchmarks and monitoring

## Contact and Resources

- **Primary Config**: `PROJECT_STATUS.md` - Detailed technical status with prioritized TODO list
- **Context Manager**: `sub-agents/context/context-manager.json` - Project structure knowledge graph
- **Dataset Docs**: `src/dataset_utils/AGENTS.md` - Dataset utilities documentation  
- **Loader Docs**: `src/dataset_utils/loaders/AGENTS.md` - Individual loader documentation
- **Testing**: `test_all_loaders.py` - Dataset loading validation

**Note**: This project is under active development. Security vulnerabilities should be addressed before production use.