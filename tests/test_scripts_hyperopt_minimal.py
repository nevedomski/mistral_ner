"""Minimal tests for scripts/hyperopt.py to improve coverage."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.hyperopt import parse_args, print_optimization_summary


def test_parse_args_minimal():
    """Test parse_args function."""
    with patch("sys.argv", ["hyperopt.py"]):
        args = parse_args()
        assert args.config == "configs/default.yaml"
        assert args.debug is False


def test_print_optimization_summary():
    """Test print_optimization_summary function."""
    mock_config = Mock()
    mock_config.hyperopt.strategy = "combined"
    mock_config.hyperopt.num_trials = 50
    mock_config.hyperopt.max_concurrent = 4
    mock_config.hyperopt.metric = "eval_f1"
    mock_config.hyperopt.mode = "max"
    mock_config.hyperopt.results_dir = "./results"
    mock_config.hyperopt.study_name = "test_study"
    mock_config.hyperopt.timeout = None
    mock_config.hyperopt.ray_address = None
    mock_config.hyperopt.search_space = {"learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-3}}

    # Should not raise any errors
    print_optimization_summary(mock_config)


def test_hyperopt_imports():
    """Test that hyperopt module imports work."""
    from scripts.hyperopt import format_search_space_summary

    # Test format function
    search_space = {
        "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-3},
        "batch_size": {"type": "choice", "choices": [16, 32, 64]},
    }

    summary = format_search_space_summary(search_space)
    assert "learning_rate" in summary
    assert "batch_size" in summary


def test_parse_args_with_options():
    """Test parse_args with command line options."""
    with patch("sys.argv", ["hyperopt.py", "--num-trials", "100", "--strategy", "optuna", "--debug"]):
        args = parse_args()
        assert args.num_trials == 100
        assert args.strategy == "optuna"
        assert args.debug is True
