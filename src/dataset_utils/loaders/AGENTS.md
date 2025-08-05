# AGENTS.md - Dataset Loaders

## Overview

This directory contains individual dataset loader implementations. Each loader is responsible for loading a specific NER dataset and converting it to a unified format.

## Dataset Loader Catalog

### Standard NER Datasets

#### 1. **conll.py** - CoNLL-2003

- **Entities**: PER, ORG, LOC, MISC
- **Size**: ~14k training samples
- **Format**: Standard CoNLL format
- **Special**: Most common benchmark dataset
- **Status**: ✅ Simple implementation, no preprocessing needed

#### 2. **ontonotes.py** - OntoNotes 5.0

- **Entities**: 18 types including PERSON, ORG, GPE, DATE, MONEY, etc.
- **Size**: Large-scale dataset
- **Format**: Uses 'tags' field (renamed to 'ner_tags')
- **Special**: Rich entity type diversity
- **Note**: May require LDC license for original data
- **⚠️ SECURITY WARNING**: Makes unvalidated HTTP requests (Line 32)
- **Risk**: SSRF attacks, DNS poisoning, network-based vulnerabilities
- **DOCDEV-TODO**: Add URL validation, timeout limits, certificate verification
- **DOCDEV-NOTE**: Uses tner/ontonotes5 preprocessed version

#### 3. **wikiner.py** - WikiNER/WikiANN

- **Entities**: PER, ORG, LOC (3 types only)
- **Size**: Multilingual, large-scale
- **Format**: Already in correct format
- **Special**: Language filtering via config_args
- **DOCDEV-NOTE**: Adds language code to config_args for filtering

#### 4. **wnut.py** - WNUT-17 Emerging Entities

- **Entities**: person, location, corporation, product, creative-work, group
- **Size**: Focused on rare/emerging entities
- **Format**: Noisy text (Twitter, Reddit)
- **Special**: Good for testing on informal text
- **Trust Remote Code**: Required

#### 5. **fewnerd.py** - Few-NERD

- **Entities**: 8 coarse types, 66 fine-grained types
- **Size**: 188k sentences
- **Format**: Supervised/few-shot settings available
- **Special**: Fine-grained entity types
- **Default**: Coarse-grained supervised setting
- **Trust Remote Code**: Required

### PII Detection Datasets

#### 6. **ai4privacy.py** - AI4Privacy PII Masking

- **Entities**: 40+ PII types (names, addresses, credit cards, etc.)
- **Size**: 65k examples
- **Format**: Complex with tokenised_unmasked_text
- **Special**: Most comprehensive PII coverage
- **⚠️ SECURITY WARNING**: Uses `ast.literal_eval()` on dataset strings (Line 196)
- **Risk**: Code injection vulnerability if dataset contains malicious code
- **DOCDEV-TODO**: Replace ast.literal_eval with safer JSON parsing or validation

#### 7. **gretel.py** - Gretel AI Synthetic PII

- **Entities**: 29 PII types focused on finance
- **Size**: Multilingual synthetic data
- **Format**: JSON spans with character offsets
- **Special**: Synthetic data, finance domain
- **DOCDEV-NOTE**: Character-to-token alignment uses simple whitespace

#### 8. **mendeley_pii.py** - Mendeley/Isotonic PII

- **Entities**: 50+ PII types
- **Size**: 200k examples
- **Format**: tokenised_text with bio_labels
- **Special**: Largest PII dataset
- **⚠️ SECURITY WARNING**: Uses `ast.literal_eval()` on dataset strings (Line 214)
- **Risk**: Code injection vulnerability if dataset contains malicious code
- **DOCDEV-TODO**: Replace ast.literal_eval with safer JSON parsing or validation
- **Memory Warning**: 200k examples - risk of OOM errors
- **DOCDEV-QUESTION**: Language filtering not implemented via config

#### 9. **bigcode_pii.py** - BigCode PII in Code

- **Entities**: PII in source code (keys, emails, names)
- **Size**: Variable (gated dataset)
- **Format**: Multiple possible field names
- **Special**: Code-specific PII detection
- **Auth**: Requires HuggingFace token
- **DOCDEV-NOTE**: Flexible field detection for various BigCode formats

## Label Mapping Patterns

### Standard Mappings

```python
# Basic 4-class mapping (CoNLL style)
"B-PER" -> "B-PER"    # Person
"B-ORG" -> "B-ORG"    # Organization  
"B-LOC" -> "B-LOC"    # Location
"B-MISC" -> "B-MISC"  # Miscellaneous
```

### PII-Specific Mappings

```python
# Financial
"B-CREDITCARDNUMBER" -> "B-CARD"
"B-IBAN" -> "B-BANK"
"B-SSN" -> "B-SSN"

# Contact
"B-EMAIL" -> "B-EMAIL"
"B-PHONE_NUMBER" -> "B-PHONE"

# Address components -> unified ADDR
"B-CITY" -> "B-ADDR"
"B-STREET" -> "B-ADDR"
"B-ZIPCODE" -> "B-ADDR"
```

### Special Cases

```python
# OntoNotes preserves rich types
"B-GPE" -> "B-GPE"       # Geo-political entity
"B-FAC" -> "B-FAC"       # Facility
"B-CARDINAL" -> "B-CARD_NUM"  # Disambiguate from CARD

# FewNERD groups fine types
"B-building" -> "B-FAC"
"B-event" -> "B-EVENT"
```

## Implementation Patterns

### Simple Loaders (no preprocessing)

- **conll.py**, **wikiner.py**, **wnut.py**, **fewnerd.py**
- Data already in correct format
- Just return examples as-is

### Field Renaming

- **ontonotes.py**: Renames 'tags' to 'ner_tags'
- Uses `rename_features` parameter

### Complex Preprocessing

- **PII datasets**: Parse JSON, align tokens, convert spans
- Handle various input formats
- Token alignment challenges

### Language Filtering Approaches

1. **Config kwargs**: `self.config_kwargs["language"] = "en"`
2. **Config args**: `self.config_args.append("en")`  
3. **No-op**: Filter in preprocessing or not supported

## Common Gotchas

1. **Token Alignment**
   - PII datasets often use character spans
   - Simple whitespace tokenization may not align properly
   - DOCDEV-TODO: Implement proper tokenizer-aware alignment

2. **String vs List Labels**
   - Some datasets return string representations of lists
   - Use `ast.literal_eval()` with try/except

3. **Missing Splits**
   - BaseDataset handles automatic splitting
   - But some datasets may have non-standard split names

4. **Authentication**
   - BigCode requires auth token
   - Set via `config_kwargs["use_auth_token"] = True`

5. **Trust Remote Code**
   - Required for: conll, wnut, fewnerd
   - Security consideration for production

## Testing Checklist

When testing a loader:

1. ✓ Can instantiate without errors
2. ✓ `load()` returns DatasetDict with train/validation/test
3. ✓ All splits have 'tokens' and 'ner_tags' features
4. ✓ Tags are in ClassLabel format
5. ✓ Label mappings cover all entity types
6. ✓ Output uses specified tagging scheme

## Performance Notes

### Memory Usage Patterns
- **Small datasets** (<50k): Load fully into memory
  - CoNLL, WNUT, WikiNER: Safe for memory loading
- **Medium datasets** (50k-100k): Monitor memory usage
  - AI4Privacy (65k): May cause issues with limited RAM
- **Large datasets** (>100k): High risk of OOM errors
  - Mendeley (200k): Requires max_samples_per_dataset or streaming
  - OntoNotes: Variable size, monitor memory usage
- **Multilingual**: Filter early to reduce memory footprint
- **Synthetic datasets**: May have quality variations

### Security Considerations
- **Gated datasets**: BigCode requires authentication tokens
- **Network dependencies**: OntoNotes fetches label mappings from web
- **Code execution risks**: ai4privacy and mendeley_pii use ast.literal_eval
- **Trust remote code**: Required for some datasets (security risk)

## Future Enhancements

1. **Standardize preprocessing patterns**
2. **Add data quality metrics**
3. **Implement streaming for large datasets**
4. **Add entity distribution statistics**
5. **Cache preprocessed datasets**
