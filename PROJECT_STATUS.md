# PROJECT_STATUS.md

## Project Overview
**mistral_ner_v2** - Complete refactoring of the original mistral_ner project for Named Entity Recognition using Mistral-7B model.

## Current Status
- **Refactoring Progress**: ~85% complete
- **Focus Area**: `src/dataset_utils/` folder
- **Last Updated**: 2025-08-05
- **Context Management**: Initialized context-manager.json for project state tracking

## Completed Components

### Dataset Utilities (90% Complete)
- ✅ **base.py**: Base class for all dataset loaders with tag scheme conversion
  - Added DOCDEV comments for critical sections
  - Refactored `_set_class_labels` method signature
  - Implemented `_apply_label_mappings` method for unified tags
  - **NEW**: Added `id2label` field to support integer-to-string tag conversion
  - **NEW**: Modified `relabel_to_class_label` to handle datasets with integer tags
- ✅ **converter.py**: Tag scheme converter supporting IO, IOB, IOB2/BIO, BIOES, BILOU schemes
  - Added DOCDEV comments explaining scheme conversions
- ✅ **All dataset loaders** in `loaders/` folder:
  - ai4privacy.py - AI4Privacy PII Masking dataset (65k examples)
  - bigcode_pii.py - BigCode PII detection in source code
  - conll.py - CoNLL-2003 standard NER dataset
  - fewnerd.py - Few-NERD fine-grained NER dataset
  - gretel.py - Gretel AI Synthetic PII Finance dataset
  - mendeley_pii.py - Mendeley/Isotonic PII dataset (200k examples)
  - ontonotes.py - OntoNotes 5.0 with 18 entity types
    - **FIXED**: Now properly converts integer tags to string labels using id2label mapping from HuggingFace
  - wikiner.py - WikiNER/WikiANN multilingual dataset
  - wnut.py - WNUT-17 Emerging Entities dataset

### Pending Components (Not Yet Refactored)
- ❌ **mixers.py**: Dataset mixing strategies (untouched) - Contains DatasetMixer class with concat, interleave, and weighted strategies
- ❌ **multi_dataset.py**: Multi-dataset loading and interleaving (untouched) - Contains MultiDatasetLoader class, depends on old architecture
- ❌ **registry.py**: Dataset registry pattern implementation (untouched) - Needs update to use new dataclass-based loaders
- ❌ **samplers.py**: Custom samplers for PyTorch DataLoader (untouched) - Contains MultiDatasetSampler for training

## Architecture Decisions

### Dataset Loader Pattern
- Each loader is a Pydantic dataclass inheriting from `BaseDataset`
- Loaders handle single datasets with configurable parameters
- Configuration can be overridden via Hydra configs
- All loaders implement `preprocess()` method for dataset-specific handling

### Tag Processing Pipeline (Three Levels)
1. **Original tags** (`ner_tags`): Dataset's native tag format
2. **Converted tags** (`single_set_tags`): Tags converted to consistent scheme (BIO by default)
3. **Unified tags** (`single_set_unified_tags`): Tags mapped to unified schema via `label_mappings`

### Tag Scheme Conversion
- Centralized in `TagSchemeConverter` class
- Automatic detection of input scheme
- Support for all major NER tagging schemes
- Conversion happens in `BaseDataset.load()` method

### Label Mappings and Unification
- Each loader defines `label_mappings` to map dataset-specific labels to unified schema
- Common unified labels: PER, ORG, LOC, MISC, DATE, TIME, MONEY, etc.
- Specialized labels for banking/PII: CARD, BANK, SSN, EMAIL, PHONE, ADDR
- Unified tags enable consistent entity types across different datasets

## Key Features Implemented
1. **Automatic train/validation/test splitting** if not present
2. **Language filtering** (especially for multilingual datasets)
3. **Tag scheme conversion** to ensure consistent output format
4. **Feature renaming** capability (e.g., OntoNotes 'tags' → 'ner_tags')
5. **ClassLabel format conversion** for HuggingFace compatibility
6. **Comprehensive label mappings** for entity type normalization
7. **Unified tags implementation** for cross-dataset consistency

## Known Issues and TODOs
1. **IterableDataset support** not fully implemented (NotImplementedError in base.py)
2. **Language filtering** implementation varies across datasets (some via config, some in preprocessing)
3. **Token alignment** in some PII datasets (e.g., Gretel) uses simple whitespace tokenization
4. **OntoNotes integer tags** - RESOLVED: Added id2label support in base class and OntoNotes loader

## Testing Status
- Test script available: `test_all_loaders.py`
- Updated to show both original tags (`ner_tags`) and converted tags (`single_set_tags`)
- Displays tag conversion examples showing how labels are mapped
- Searches for examples with entity tags (not just 'O' tags) for better demonstration
- Two-level validation:
  - **Dataset schema**: Compares converted tags with mapping keys (validates dataset output)
  - **Unified schema**: Placeholder for comparing unified tags with mapping values (TODO)
- All loaders can be instantiated and basic loading tested
- Some datasets require authentication (BigCode, potentially OntoNotes)

## Configuration Integration

- Datasets configured via Hydra YAML files in `configs/data/`
- Each loader supports config_args and config_kwargs for HuggingFace dataset loading
- Output tagging scheme configurable (default: BIO)

## Linear Issues

*[To be populated with Linear issue links and descriptions]*

## Recent Changes (2025-01-22)

### Fixed OntoNotes Dataset Integer Tag Issue
- **Problem**: OntoNotes dataset from `tner/ontonotes5` uses integer tags (0-36) without providing label names
- **Solution**: 
  - Added `id2label: Callable[[list[int]], list[str]] | None` field to BaseDataset class
  - Modified `relabel_to_class_label` to detect integer tags and use id2label conversion
  - Implemented `id2label` method in OntoNotesDataset that fetches mapping from official dataset label.json
  - URL: https://huggingface.co/datasets/tner/ontonotes5/raw/main/dataset/label.json
- **Result**: OntoNotes tags now properly convert (e.g., 4 → "B-PERSON" → "B-PER")

## Next Steps

1. Complete refactoring of untouched files (mixers.py, multi_dataset.py, registry.py, samplers.py)
2. Add comprehensive unit tests for all loaders
3. Implement proper IterableDataset support
4. Standardize language filtering across all datasets
5. Add performance benchmarks for dataset loading
6. **TODO**: Add persistent cache for:
   - Loaded datasets to avoid re-downloading/processing
   - OntoNotes id2label mapping (currently uses `@lru_cache` but not persistent across runs)

## Notes for Future Development

- When adding new loaders, follow the existing pattern in loaders/
- Ensure label_mappings are comprehensive for entity type normalization
- Test with `test_all_loaders.py` before committing
- Update registry.py when adding new datasets
- Consider memory usage for large datasets (use IterableDataset when appropriate)

## Recent Changes (2025-08-05)

### Comprehensive Project Review Conducted
- **Parallel Analysis**: Used multiple specialized agents (architecture, code quality, dataset implementation, documentation)
- **Key Findings**:
  - Architecture: Excellent foundations with Hydra + Pydantic, but 15% incomplete
  - Critical Issues: Multi-dataset training broken, security vulnerabilities, memory management issues
  - Documentation: Missing README.md, no unit tests, insufficient user guides

### Critical Security Vulnerabilities Identified
- **ast.literal_eval on untrusted input**: Found in ai4privacy.py and mendeley_pii.py loaders
  - Risk: Code injection if dataset is compromised
  - Solution: Replace with json.loads and input validation
- **Unvalidated HTTP requests**: OntoNotes loader lacks timeout, SSL verification
  - Risk: SSRF attacks, DNS poisoning
  - Solution: Add URL validation, certificate verification, proper timeouts
- **Missing input validation**: Critical paths lack proper validation (base.py id2label)
  - Risk: Runtime crashes, data corruption
  - Solution: Validate callable functions and parameters

### Architectural Issues Documented
- **Multi-Dataset Implementation Incomplete**: Core feature non-functional
  - Missing: _mixers.py, _multi_dataset.py, _samplers.py, registry.py
  - Impact: Cannot train on multiple datasets as advertised
- **Configuration Coupling**: Cross-module state modification in validators
  - Location: config.py setup_wandb validator
  - Impact: Tight coupling between configuration sections
- **Memory Management**: No streaming support for large datasets
  - Risk: OOM crashes with large datasets
  - Missing: IterableDataset support, persistent caching

### DOCDEV Comments Added
- **src/data.py**: Edge case validation for tokenization
- **src/config_datasets.py**: Critical incomplete multi-dataset feature
- **src/config.py**: Configuration coupling issue
- **Existing comments verified**: Security vulnerabilities in loaders already documented

## Recent Changes (2025-07-24)

### Fixed Linting Errors in conf_hyperopt.py
- **Problem**: Multiple linting errors in hyperparameter optimization configuration
  - `IntChoiceConfig.choices` was incorrectly defined as `ClassVar[list[int]]`
  - `resources_per_trial` field was trying to instantiate `ResourceConfig` directly
  - Missing pydantic `Field` import for proper field factory usage
- **Solution**:
  - Added `from pydantic import Field` import
  - Changed `choices: ClassVar[list[int]] = []` to `choices: list[int] = Field(default_factory=list)`
  - Fixed `resources_per_trial` to use `Field(default_factory=lambda: ResourceConfig(cpu=1.0, gpu=1.0))`
- **Result**: All mypy and ruff checks now pass successfully

### Updated Configuration Files Documentation and Fixed Issues
- **Fixed Logger Names**: Changed logger in `conf_hyperopt.py` from "mistral_ner.model" to "mistral_ner.hyperopt"
- **Updated Default Quantization**: Set `load_in_8bit=True` as default in `conf_model.py` for better out-of-box experience
- **Added Quantization Validation**: Added `@model_validator` in `conf_model.py` to ensure only one quantization method is enabled
- **Fixed Search Space**: Removed `ClassVar` from `search_space` in OptimizationConfig and used proper factory function
- **Updated All Docstrings**: Fixed docstrings in all three config files to accurately reflect their purpose and fields

### Fixed and Updated conf_training.py
- **Fixed Class Name**: Changed incorrect `LoggingConfig` to `TrainingConfig`
- **Updated Module Docstring**: Changed from generic "Base class for NER training settings" to specific description
- **Comprehensive Docstring**: Added detailed documentation for all 40+ training parameters
- **Added Validation**: Implemented `@model_validator` with checks for:
  - Mixed precision conflicts (fp16 vs bf16)
  - Loss function parameter validation
  - Effective batch size warnings
  - Strategy consistency warnings
- **Added Type Annotations**: Fixed missing type annotation for `lr_scheduler_kwargs`
- **Enhanced Comments**: Added inline comments for mixed precision settings
- **Result**: All mypy and ruff checks pass successfully

### Updated config.py and conf_datasets.py
- **config.py**:
  - Cleaned up large section of commented-out code (old DatasetConfig and DataConfig classes)
  - Updated Config class docstring with comprehensive documentation
  - Added examples showing different initialization methods
  - Documented the setup_wandb validator behavior
- **conf_datasets.py**:
  - Fixed module docstring to "Multi-dataset configuration for Mistral NER fine-tuning"
  - Changed logger name from "mistral_ner.multi_datasets" to "mistral_ner.datasets"
  - Completely rewrote MultiDatasetConfig docstring with accurate field descriptions
  - Improved prepare_datasets function with better error handling and documentation
  - Added field_validator for datasets that properly handles configuration transformation
- **Result**: Both files pass ruff checks and formatting

## TODO List (Prioritized)

### Critical Security Issues (HIGH PRIORITY - Immediate Action Required)
- [ ] **DOCDEV-TODO**: Replace `ast.literal_eval()` with json.loads in ai4privacy.py (line 193)
- [ ] **DOCDEV-TODO**: Replace `ast.literal_eval()` with json.loads in mendeley_pii.py (line 211) 
- [ ] **DOCDEV-TODO**: Add URL validation, timeout, and SSL verification to ontonotes.py HTTP requests
- [ ] **DOCDEV-TODO**: Add input validation for id2label function in base.py (line 155)

### Architectural Issues (HIGH PRIORITY)
- [ ] **DOCDEV-TODO**: Complete multi-dataset training implementation (config_datasets.py line 138)
- [ ] **DOCDEV-TODO**: Refactor configuration coupling in config.py setup_wandb validator
- [ ] **DOCDEV-TODO**: Implement IterableDataset support in base.py for streaming large datasets
- [ ] **DOCDEV-TODO**: Add persistent cache for loaded datasets and OntoNotes id2label mapping

### Incomplete Components (MEDIUM PRIORITY)
- [ ] **DOCDEV-TODO**: Refactor _mixers.py to match new dataclass loader pattern
- [ ] **DOCDEV-TODO**: Update _multi_dataset.py to work with new loader architecture  
- [ ] **DOCDEV-TODO**: Update registry.py to use new dataclass-based loaders
- [ ] **DOCDEV-TODO**: Implement _samplers.py for multi-dataset training

### Code Quality (MEDIUM PRIORITY)
- [ ] **DOCDEV-TODO**: Add validation for empty word_ids and label length mismatches in data.py
- [ ] **DOCDEV-TODO**: Standardize language filtering approach across all dataset loaders
- [ ] **DOCDEV-TODO**: Implement proper subword tokenization alignment for PII datasets
- [ ] **DOCDEV-TODO**: Add conversion cycle detection and depth limits in TagSchemeConverter

### Testing and Documentation (LOW PRIORITY)
- [ ] Add comprehensive unit tests achieving 90%+ coverage
- [ ] Create README.md with installation and usage instructions
- [ ] Add performance benchmarks for dataset loading
- [ ] Document memory usage patterns for different dataset sizes
- [ ] Create user guide for multi-dataset training

### Infrastructure Improvements
- [ ] Implement plugin-based dataset discovery system
- [ ] Add memory monitoring and automatic cleanup
- [ ] Create GitHub Actions CI/CD pipeline
- [ ] Set up automated security scanning

## Context Management Updates (2025-08-05)

### Context Manager Initialized
- Created `sub-agents/context/context-manager.json` with complete project structure
- Established directory for AI agent context management
- Set up knowledge graph for project state tracking

### DOCDEV Comments Status
- Total DOCDEV comments in codebase: 54
- DOCDEV-TODO: 31 (action items tracked above)
- DOCDEV-NOTE: 20 (documentation/explanations)
- DOCDEV-QUESTION: 3 (design decisions to review)

### Next Context Management Steps
1. Set up incremental update workflow for context-manager.json
2. Create agent activity logging system
3. Implement query interface for project structure
4. Add persistent caching for frequently accessed data