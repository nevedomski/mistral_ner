#!/usr/bin/env python3
"""Simple test runner without pytest."""

import sys
import traceback
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Test results
passed = 0
failed = 0
errors = []


def run_test(test_func, test_name):
    """Run a single test function."""
    global passed, failed, errors
    
    try:
        print(f"Running {test_name}...", end=" ")
        test_func()
        print("✓ PASSED")
        passed += 1
    except AssertionError as e:
        print("✗ FAILED")
        failed += 1
        errors.append({
            "test": test_name,
            "error": str(e),
            "type": "assertion"
        })
    except Exception as e:
        print("✗ ERROR")
        failed += 1
        errors.append({
            "test": test_name,
            "error": str(e),
            "type": "exception",
            "traceback": traceback.format_exc()
        })


def test_imports():
    """Test that all modules can be imported."""
    print("\n=== Testing Imports ===")
    
    modules = [
        "src.config",
        "src.utils",
        "src.data",
        "src.model",
        "src.evaluation",
        "src.training"
    ]
    
    for module in modules:
        def test():
            __import__(module)
        run_test(test, f"import {module}")


def test_config():
    """Test configuration functionality."""
    print("\n=== Testing Configuration ===")
    
    from src.config import Config, DataConfig
    
    def test_default_config():
        config = Config()
        assert config.model.model_name == "mistralai/Mistral-7B-v0.3"
        assert config.model.num_labels == 9
        assert config.model.load_in_8bit == True
        assert config.data.max_length == 256
        assert config.training.num_train_epochs == 5
    
    def test_data_config():
        data_config = DataConfig()
        assert len(data_config.label_names) == 9
        assert data_config.id2label[0] == "O"
        assert data_config.label2id["O"] == 0
    
    def test_config_update():
        config = Config()
        class Args:
            model_name = "test-model"
            learning_rate = 1e-3
            use_wandb = False
        
        args = Args()
        config.update_from_args(args)
        assert config.model.model_name == "test-model"
        assert config.training.learning_rate == 1e-3
        assert config.logging.use_wandb == False
    
    run_test(test_default_config, "Default configuration")
    run_test(test_data_config, "Data configuration")
    run_test(test_config_update, "Configuration update from args")


def test_utils():
    """Test utility functions."""
    print("\n=== Testing Utils ===")
    
    from src.utils import estimate_memory_usage, detect_mixed_precision_support
    
    def test_memory_estimation():
        estimate = estimate_memory_usage(
            model_size_gb=14.0,
            batch_size=4,
            sequence_length=256,
            use_8bit=True,
            use_lora=True
        )
        assert "model_memory_gb" in estimate
        assert "total_memory_gb" in estimate
        assert estimate["model_memory_gb"] < 14.0  # Should be reduced with 8-bit
    
    def test_mixed_precision():
        support = detect_mixed_precision_support()
        assert isinstance(support, dict)
        assert "fp16" in support
        assert "bf16" in support
        assert all(isinstance(v, bool) for v in support.values())
    
    run_test(test_memory_estimation, "Memory estimation")
    run_test(test_mixed_precision, "Mixed precision detection")


def test_data():
    """Test data functionality."""
    print("\n=== Testing Data Module ===")
    
    from src.data import create_sample_dataset, get_label_list
    
    def test_sample_dataset():
        dataset = create_sample_dataset(size=10)
        assert "train" in dataset
        assert "validation" in dataset
        assert "test" in dataset
        assert len(dataset["train"]) == 10
        assert len(dataset["validation"]) == 2
    
    def test_label_list():
        dataset = create_sample_dataset(size=5)
        labels = get_label_list(dataset)
        assert isinstance(labels, list)
        assert len(labels) == 9
        assert labels[0] == "O"
    
    run_test(test_sample_dataset, "Sample dataset creation")
    run_test(test_label_list, "Label list extraction")


def main():
    """Run all tests."""
    print("Running Mistral NER Tests")
    print("=" * 50)
    
    # Run test suites
    test_imports()
    test_config()
    test_utils()
    test_data()
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests Summary: {passed} passed, {failed} failed")
    
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"\n{error['test']}:")
            print(f"  Type: {error['type']}")
            print(f"  Error: {error['error']}")
            if 'traceback' in error:
                print(f"  Traceback:\n{error['traceback']}")
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()