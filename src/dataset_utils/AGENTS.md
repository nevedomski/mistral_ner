# AGENTS.md - Dataset Utilities

## Overview
This directory contains the core dataset loading and processing infrastructure for the Mistral NER v2 project. It implements a flexible, extensible system for loading various NER datasets with unified processing pipelines.

## Architecture

### Core Components

1. **base.py** (90% refactored)
   - Abstract base class `BaseDataset` that all loaders inherit from
   - Handles common operations: loading from HuggingFace Hub, splitting, tag conversion, unification
   - Key features:
     - Automatic train/validation/test split creation
     - Tag scheme conversion via `TagSchemeConverter`
     - Label mapping to unified schema via `_apply_label_mappings`
     - ClassLabel format normalization
     - Dataset validation
   - DOCDEV-TODO: Add support for IterableDataset (currently raises NotImplementedError)

2. **converter.py** (80% refactored)
   - `TagSchemeConverter` class for converting between NER tagging schemes
   - Supports: IO, IOB, IOB2/BIO, BIOES, BILOU
   - Automatic scheme detection
   - DOCDEV-NOTE: IOB vs IOB2 detection is heuristic-based, may need refinement

3. **registry.py** (untouched)
   - Registry pattern for dynamic dataset loader discovery
   - Avoids hardcoded imports
   - DOCDEV-TODO: Needs refactoring to match new loader pattern

4. **multi_dataset.py** (untouched)
   - Multi-dataset loading and interleaving
   - Weighted sampling strategies
   - DOCDEV-TODO: Needs refactoring to work with new loader architecture

5. **mixers.py** (untouched)
   - Dataset mixing strategies
   - DOCDEV-TODO: Needs implementation

6. **samplers.py** (untouched)
   - Custom sampling strategies
   - DOCDEV-TODO: Needs implementation

### Loader Pattern

All dataset loaders follow this pattern:

```python
@dataclass
class CustomDataset(BaseDataset):
    name: str = "dataset_name"
    developer: str = "org_name"
    label_mappings: ClassVar[dict[str, str] | None] = {
        "O": "O",
        "B-CUSTOM": "B-PER",  # Map to unified schema
        # ...
    }
    
    def preprocess(self, examples: dict[str, Any]) -> dict[str, Any]:
        # Dataset-specific preprocessing
        return {"tokens": [...], "ner_tags": [...]}
```

## Critical Implementation Details

### Label Mapping Strategy
- Each dataset maps its specific labels to a unified schema
- Common unified labels:
  - **PER**: Person names
  - **ORG**: Organizations, companies
  - **LOC**: General locations
  - **GPE**: Geo-political entities (countries, cities, states)
  - **FAC**: Facilities (buildings, airports, etc.)
  - **MISC**: Miscellaneous entities
  - **Banking/PII specific**:
    - **CARD**: Credit card information
    - **BANK**: Bank accounts, IBAN, routing numbers
    - **SSN**: Social Security Numbers
    - **EMAIL**: Email addresses
    - **PHONE**: Phone numbers
    - **ADDR**: Street addresses

### Tag Scheme Conversion Flow
1. Dataset loads with original tags (stored in `ner_tags`)
2. `preprocess()` converts to standard format
3. `relabel_to_class_label()` ensures ClassLabel format
4. `_convert_tag_scheme()` converts to output scheme (default: BIO)
5. Converted tags are stored in `single_set_tags` for single-dataset training
6. `_apply_label_mappings()` creates unified tags using label_mappings
7. Unified tags are stored in `single_set_unified_tags`
8. Final validation ensures consistency

**Important**: 
- `ner_tags`: Contains original tags from the dataset
- `single_set_tags`: Contains tags after conversion to the desired scheme
- `single_set_unified_tags`: Contains tags mapped to the unified schema

### Configuration Integration
- Loaders accept configuration via Pydantic fields
- Can be overridden by Hydra configs in `configs/data/`
- Key configurable parameters:
  - `output_tagging_scheme`: Target tag scheme
  - `filter_english`: Language filtering
  - `max_samples_per_dataset`: Memory management
  - `config_args/config_kwargs`: HuggingFace dataset parameters

## Critical Security Vulnerabilities ⚠️

1. **ast.literal_eval() Security Issues**
   - **ai4privacy.py (Line 196)**: Uses `ast.literal_eval()` on untrusted dataset strings
   - **mendeley_pii.py (Line 214)**: Uses `ast.literal_eval()` on untrusted dataset strings
   - **Risk**: Code injection if malicious strings in dataset
   - **DOCDEV-TODO**: Replace with safer JSON parsing or explicit validation

2. **Network Requests Without Proper Validation**
   - **ontonotes.py (Line 32)**: Makes unvalidated HTTP requests to HuggingFace Hub
   - **Risk**: SSRF attacks, DNS poisoning, network-based attacks
   - **DOCDEV-TODO**: Add URL validation, timeout limits, and certificate verification

3. **Missing Input Validation**
   - **base.py (Line 153-159)**: No validation of id2label function parameters
   - **Risk**: Potential for arbitrary code execution or data corruption
   - **DOCDEV-TODO**: Add strict input validation for all external data

## Memory Management Issues

1. **Memory Leak Risks**
   - **base.py (Line 196-223)**: Large datasets loaded entirely into memory without cleanup
   - **Risk**: OOM crashes, especially with multi-dataset training
   - **DOCDEV-TODO**: Implement proper memory management and streaming

2. **Large Dataset Handling**
   - AI4Privacy: 65k examples, Mendeley: 200k examples
   - No automatic memory optimization
   - **DOCDEV-TODO**: Implement IterableDataset support for large datasets

## Incomplete Components (High Risk)

1. **_mixers.py**: Partial implementation, may cause runtime errors
2. **_multi_dataset.py**: Uses old architecture, incompatible with current loaders
3. **_samplers.py**: Incomplete multi-dataset sampling logic
4. **registry.py**: Outdated pattern, needs refactoring for new dataclass loaders

## Known Issues and Gotchas

1. **Language Filtering Inconsistency**
   - Some datasets filter via `config_kwargs["language"]`
   - Others filter in preprocessing
   - WikiNER uses config_args
   - **DOCDEV-TODO**: Standardize language filtering approach

2. **Token Alignment in PII Datasets**
   - Gretel and similar datasets use character spans
   - Simple whitespace tokenization may misalign
   - **Risk**: Incorrect entity boundaries, training data corruption
   - **DOCDEV-TODO**: Implement proper subword tokenization alignment

3. **Authentication Requirements**
   - BigCode datasets are gated
   - OntoNotes may require LDC license
   - Handle via `config_kwargs["use_auth_token"]`

4. **Configuration Coupling Issues**
   - **config_datasets.py**: Hardcoded dataset imports create tight coupling
   - Difficult to add new datasets without modifying core files
   - **DOCDEV-TODO**: Implement plugin-based registry system

5. **Potential Infinite Loops**
   - **converter.py (Line 117-119)**: Chained lambda conversions may create cycles
   - **Risk**: Application hang, resource exhaustion
   - **DOCDEV-TODO**: Add conversion cycle detection and depth limits

## Testing and Validation

- Use `test_all_loaders.py` to test dataset loading
  - Shows original, converted, and unified tags side-by-side
  - Displays tag conversion examples with actual entity tags (not just 'O')
  - Searches first 100 examples to find meaningful entity examples
  - Two-level validation:
    - **Dataset schema**: Converted tags match mapping keys (validates dataset output)
    - **Unified schema**: Unified tags match mapping values (validates cross-dataset consistency)
- Each loader should handle missing splits gracefully
- Validation ensures:
  - Required features: tokens, ner_tags, single_set_tags, single_set_unified_tags
  - Consistent lengths
  - ClassLabel format
  - Converted tags match the expected dataset schema
  - Unified tags enable consistent entity types across datasets

## Adding New Datasets

1. Create new file in `loaders/` following the pattern
2. Define `label_mappings` for entity normalization
3. Implement `preprocess()` if format conversion needed
4. Update `registry.py` to include new loader
5. Create config in `configs/data/` if needed
6. Test with `test_all_loaders.py`

## Performance Considerations

- Datasets are loaded fully into memory by default
- Large datasets may require:
  - Setting `max_samples_per_dataset`
  - Implementing IterableDataset support
  - Using batch processing in preprocessing

## Future Improvements

1. **Streaming Support**: Implement IterableDataset for large datasets
2. **Unified Tokenization**: Standardize tokenization across all datasets
3. **Advanced Sampling**: Implement balanced sampling strategies
4. **Caching**: Add processed dataset caching
5. **Validation Suite**: Comprehensive testing for label mappings