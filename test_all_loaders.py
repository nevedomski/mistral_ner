#!/usr/bin/env python3
"""Temporary test script to load all dataset loaders and test their load() method.

This script now shows both:
- 'ner_tags': Original tags from the dataset
- 'single_set_tags': Tags after conversion to the desired scheme (BIO by default)

The tag conversion example demonstrates how entity tags are transformed based on
the label_mappings defined in each loader.
"""

import sys
import traceback
from typing import Any

# Import all loaders
from src.dataset_utils.loaders.ai4privacy import AI4PrivacyDataset
from src.dataset_utils.loaders.bigcode_pii import BigCodePIIDataset
from src.dataset_utils.loaders.conll import CoNLLDataset
from src.dataset_utils.loaders.fewnerd import FewNERDDataset
from src.dataset_utils.loaders.gretel import GretelPIIDataset
from src.dataset_utils.loaders.mendeley_pii import MendeleyPIIDataset
from src.dataset_utils.loaders.ontonotes import OntoNotesDataset
from src.dataset_utils.loaders.wikiner import WikiNERDataset
from src.dataset_utils.loaders.wnut import WNUTDataset


def test_loader(loader_class: type, name: str) -> dict[str, Any]:
    """Test a single loader.

    Args:
        loader_class: The loader class to test
        name: Name of the loader for display

    Returns:
        Dict with test results
    """
    print(f"\n{'=' * 60}")
    print(f"Testing {name}")
    print("=" * 60)

    result = {"name": name, "success": False, "error": None, "dataset_info": None}

    try:
        # Initialize the loader
        print(f"Initializing {name}...")
        loader = loader_class()
        print("✓ Initialized successfully")
        print(f"  - Full name: {loader.full_name}")
        print(f"  - Config args: {loader.config_args}")
        print(f"  - Config kwargs: {loader.config_kwargs}")

        # Try to load the dataset
        print("\nLoading dataset...")
        dataset = loader.load()

        # If successful, get some info
        print("✓ Dataset loaded successfully!")

        dataset_info = {
            "splits": list(dataset.keys()),
            "train_size": len(dataset.get("train", [])),
            "validation_size": len(dataset.get("validation", [])),
            "test_size": len(dataset.get("test", [])),
        }

        # Check features
        if "train" in dataset and len(dataset["train"]) > 0:
            dataset_info["features"] = list(dataset["train"].features.keys())

            # Find an example with entity tags (not just 'O' tags)
            example_idx = 0
            max_examples_to_check = min(100, len(dataset["train"]))
            found_entities = False

            for idx in range(max_examples_to_check):
                tags = dataset["train"][idx].get("ner_tags", [])
                # Check if this example has any non-O tags
                if isinstance(tags[0], int):
                    # For integer tags, check if any are non-zero (assuming O is 0)
                    if any(t != 0 for t in tags[:20]):  # Check first 20 tags
                        example_idx = idx
                        found_entities = True
                        break
                else:
                    # For string tags, check if any are not 'O'
                    if any(t != "O" for t in tags[:20]):
                        example_idx = idx
                        found_entities = True
                        break

            if not found_entities:
                print("  Note: No entity tags found in first 100 examples, using first example")

            dataset_info["sample_tokens"] = dataset["train"][example_idx].get("tokens", [])[:10]

            # Get original tags, converted tags, and unified tags
            dataset_info["sample_original_tags"] = dataset["train"][example_idx].get("ner_tags", [])[:10]
            dataset_info["sample_converted_tags"] = dataset["train"][example_idx].get("single_set_tags", [])[:10]
            dataset_info["sample_unified_tags"] = dataset["train"][example_idx].get("single_set_unified_tags", [])[:10]

            # Show original tags in string format if they are integers
            if isinstance(dataset_info["sample_original_tags"][0], int):
                if "ner_tags" in dataset["train"].features:
                    feature = dataset["train"].features["ner_tags"]
                    if hasattr(feature, "feature") and hasattr(feature.feature, "names"):
                        tag_names = feature.feature.names
                        string_tags = [tag_names[t] if t < len(tag_names) else f"UNKNOWN_{t}" for t in dataset_info["sample_original_tags"]]
                        dataset_info["sample_original_tags_str"] = string_tags

            # Show converted tags in string format if they are integers
            if isinstance(dataset_info["sample_converted_tags"][0], int):
                if "single_set_tags" in dataset["train"].features:
                    feature = dataset["train"].features["single_set_tags"]
                    if hasattr(feature, "feature") and hasattr(feature.feature, "names"):
                        tag_names = feature.feature.names
                        string_tags = [
                            tag_names[t] if t < len(tag_names) else f"UNKNOWN_{t}" for t in dataset_info["sample_converted_tags"]
                        ]
                        dataset_info["sample_converted_tags_str"] = string_tags

            # Show unified tags in string format if they are integers
            if isinstance(dataset_info["sample_unified_tags"][0], int):
                if "single_set_unified_tags" in dataset["train"].features:
                    feature = dataset["train"].features["single_set_unified_tags"]
                    if hasattr(feature, "feature") and hasattr(feature.feature, "names"):
                        tag_names = feature.feature.names
                        string_tags = [tag_names[t] if t < len(tag_names) else f"UNKNOWN_{t}" for t in dataset_info["sample_unified_tags"]]
                        dataset_info["sample_unified_tags_str"] = string_tags

            # Check if dataset contains all expected NER tags (using converted tags)
            if hasattr(loader_class, "label_mappings") and loader_class.label_mappings:
                # Get unique tags from first few examples - check both original and converted
                unique_original_tags = set()
                unique_converted_tags = set()
                for i in range(min(100, len(dataset["train"]))):
                    # Original tags
                    original_tags = dataset["train"][i].get("ner_tags", [])
                    if isinstance(original_tags[0], str):
                        unique_original_tags.update(original_tags)
                    else:
                        # If tags are integers, we need the label names
                        if "ner_tags" in dataset["train"].features:
                            feature = dataset["train"].features["ner_tags"]
                            if hasattr(feature, "feature") and hasattr(feature.feature, "names"):
                                tag_names = feature.feature.names
                                unique_original_tags.update(tag_names[tag] for tag in original_tags if tag < len(tag_names))

                    # Converted tags
                    converted_tags = dataset["train"][i].get("single_set_tags", [])
                    if isinstance(converted_tags[0], str):
                        unique_converted_tags.update(converted_tags)
                    else:
                        # If tags are integers, we need the label names
                        if "single_set_tags" in dataset["train"].features:
                            feature = dataset["train"].features["single_set_tags"]
                            if hasattr(feature, "feature") and hasattr(feature.feature, "names"):
                                tag_names = feature.feature.names
                                unique_converted_tags.update(tag_names[tag] for tag in converted_tags if tag < len(tag_names))

                # Check coverage - compare mapping keys (dataset schema) with converted tags
                expected_tags = set(loader_class.label_mappings.keys())
                print("\n  Label mapping coverage:")
                print(f"    - Expected tags in mappings (keys): {len(expected_tags)}")
                print(f"    - Unique original tags found: {len(unique_original_tags)}")
                print(f"    - Unique converted tags found: {len(unique_converted_tags)}")
                print(f"    - Converted tags match expected unified tags: {unique_converted_tags == expected_tags}")
                if unique_converted_tags != expected_tags:
                    missing_in_data = expected_tags - unique_converted_tags
                    extra_in_data = unique_converted_tags - expected_tags
                    if missing_in_data:
                        print(f"    - Expected tags not found in converted data: {missing_in_data}")
                    if extra_in_data:
                        print(f"    - Converted tags not in expected tags: {extra_in_data}")

                # Check coverage - compare mapping VALUES (unified schema) with actual unified tags
                unique_unified_tags = set()
                for i in range(min(100, len(dataset["train"]))):
                    # Unified tags
                    unified_tags = dataset["train"][i].get("single_set_unified_tags", [])
                    if isinstance(unified_tags[0], str):
                        unique_unified_tags.update(unified_tags)
                    else:
                        # If tags are integers, we need the label names
                        if "single_set_unified_tags" in dataset["train"].features:
                            feature = dataset["train"].features["single_set_unified_tags"]
                            if hasattr(feature, "feature") and hasattr(feature.feature, "names"):
                                tag_names = feature.feature.names
                                unique_unified_tags.update(tag_names[tag] for tag in unified_tags if tag < len(tag_names))

                expected_unified_tags = set(loader_class.label_mappings.values())
                print("\n  Unified schema validation:")
                print(f"    - Expected unified tags (values): {len(expected_unified_tags)}")
                print(f"    - Unique unified tags found: {len(unique_unified_tags)}")
                print(f"    - Unified tags match expected unified tags: {unique_unified_tags == expected_unified_tags}")
                if unique_unified_tags != expected_unified_tags:
                    missing_in_data = expected_unified_tags - unique_unified_tags
                    extra_in_data = unique_unified_tags - expected_unified_tags
                    if missing_in_data:
                        print(f"    - Expected unified tags not found in unified data: {missing_in_data}")
                    if extra_in_data:
                        print(f"    - Unified tags not in expected unified tags: {extra_in_data}")

        result["success"] = True
        result["dataset_info"] = dataset_info

        print("\nDataset info:")
        print(f"  - Splits: {dataset_info['splits']}")
        print(f"  - Train size: {dataset_info['train_size']}")
        print(f"  - Validation size: {dataset_info['validation_size']}")
        print(f"  - Test size: {dataset_info['test_size']}")
        if "features" in dataset_info:
            print(f"  - Features: {dataset_info['features']}")
            print(f"  - Sample tokens: {dataset_info['sample_tokens']}")
            print(f"  - Original tags (ner_tags): {dataset_info['sample_original_tags']}")
            if "sample_original_tags_str" in dataset_info:
                print(f"  - Original tags (string): {dataset_info['sample_original_tags_str']}")
            print(f"  - Converted tags (single_set_tags): {dataset_info['sample_converted_tags']}")
            if "sample_converted_tags_str" in dataset_info:
                print(f"  - Converted tags (string): {dataset_info['sample_converted_tags_str']}")
            print(f"  - Unified tags (single_set_unified_tags): {dataset_info['sample_unified_tags']}")
            if "sample_unified_tags_str" in dataset_info:
                print(f"  - Unified tags (string): {dataset_info['sample_unified_tags_str']}")

            # Show tag conversion example if all string versions are available
            if (
                "sample_original_tags_str" in dataset_info
                and "sample_converted_tags_str" in dataset_info
                and "sample_unified_tags_str" in dataset_info
            ):
                print("\n  Tag Conversion Example:")
                print("  Tokens | Original Tag | Converted Tag | Unified Tag")
                print("  " + "-" * 60)
                for i in range(min(10, len(dataset_info["sample_tokens"]))):
                    token = dataset_info["sample_tokens"][i]
                    orig_tag = dataset_info["sample_original_tags_str"][i] if i < len(dataset_info["sample_original_tags_str"]) else "?"
                    conv_tag = dataset_info["sample_converted_tags_str"][i] if i < len(dataset_info["sample_converted_tags_str"]) else "?"
                    unif_tag = dataset_info["sample_unified_tags_str"][i] if i < len(dataset_info["sample_unified_tags_str"]) else "?"
                    print(f"  {token:<12} | {orig_tag:<12} | {conv_tag:<13} | {unif_tag}")

    except Exception as e:
        result["error"] = str(e)
        print(f"\n✗ Error loading dataset: {e}")
        print("\nTraceback:")
        traceback.print_exc()

    return result


def main():
    """Run tests on all loaders."""
    print("Testing all dataset loaders...")
    print(f"Python: {sys.version}")

    # List of all loaders to test
    loaders = [
        (AI4PrivacyDataset, "AI4Privacy"),
        (BigCodePIIDataset, "BigCodePII"),
        (CoNLLDataset, "CoNLL-2003"),
        (FewNERDDataset, "FewNERD"),
        (GretelPIIDataset, "GretelPII"),
        (MendeleyPIIDataset, "MendeleyPII"),
        (OntoNotesDataset, "OntoNotes"),
        (WikiNERDataset, "WikiNER"),
        (WNUTDataset, "WNUT-17"),
    ]

    results = []

    # Test each loader
    for loader_class, name in loaders:
        result = test_loader(loader_class, name)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print(f"\nTotal loaders tested: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed loaders:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['name']}: {r['error']}")

    print("\nNote: Some datasets may require authentication or have download restrictions.")
    print("This is expected behavior for certain datasets.")


if __name__ == "__main__":
    main()
