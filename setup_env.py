#!/usr/bin/env python3
"""
Environment detection and setup script for mistral-ner project.
Automatically detects CUDA availability and version, then provides appropriate installation commands.
"""

import subprocess
import sys
import platform
import os


def get_cuda_version():
    """Detect CUDA version from nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            # Parse CUDA version from nvidia-smi output
            for line in result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    cuda_str = line.split('CUDA Version:')[1].strip().split()[0]
                    version = float(cuda_str.split('.')[0] + '.' + cuda_str.split('.')[1])
                    return version
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    # Try nvcc as fallback
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    version_str = line.split('release')[1].strip().split(',')[0]
                    version = float(version_str)
                    return version
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    return None


def check_pytorch_cuda():
    """Check if PyTorch is installed and has CUDA support."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return None


def get_environment_info():
    """Gather environment information."""
    info = {
        'platform': platform.system(),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
        'cuda_version': get_cuda_version(),
        'pytorch_cuda': check_pytorch_cuda(),
        'is_conda': 'CONDA_DEFAULT_ENV' in os.environ,
        'conda_env': os.environ.get('CONDA_DEFAULT_ENV', None)
    }
    return info


def get_installation_command(env_info):
    """Generate appropriate installation command based on environment."""
    cuda_version = env_info['cuda_version']
    
    if cuda_version is None:
        # CPU-only installation
        return "uv pip install .[cpu]", "cpu"
    elif cuda_version >= 12.0:
        # CUDA 12.x
        return "uv pip install .[cuda12]", "cuda12"
    elif cuda_version >= 11.0:
        # CUDA 11.x
        return "uv pip install .[cuda11]", "cuda11"
    else:
        # Generic CUDA (older versions)
        return "uv pip install .[cuda]", "cuda"


def main():
    print("🔍 Detecting environment configuration...")
    print("-" * 50)
    
    env_info = get_environment_info()
    
    print(f"Platform: {env_info['platform']}")
    print(f"Python version: {env_info['python_version']}")
    print(f"Is Conda environment: {env_info['is_conda']}")
    if env_info['is_conda']:
        print(f"Conda environment name: {env_info['conda_env']}")
    
    if env_info['cuda_version']:
        print(f"CUDA version detected: {env_info['cuda_version']}")
    else:
        print("CUDA not detected - will use CPU-only configuration")
    
    if env_info['pytorch_cuda'] is not None:
        print(f"PyTorch CUDA support: {'✓ Enabled' if env_info['pytorch_cuda'] else '✗ Disabled'}")
    
    print("-" * 50)
    
    try:
        install_cmd, env_type = get_installation_command(env_info)
        
        print(f"\n📦 Recommended installation command:")
        print(f"\n    {install_cmd}\n")
        
        if env_info['is_conda']:
            print("💡 For conda environments, you can also use:")
            print(f"    conda env create -f environment-{env_type}.yml")
            print("    conda activate mistral-ner")
    except RuntimeError as e:
        print(f"\n❌ {e}")
        print("\nThis project requires a GPU with CUDA support.")
        print("Please ensure you have:")
        print("  1. NVIDIA GPU installed")
        print("  2. NVIDIA drivers installed")
        print("  3. CUDA toolkit installed")
        sys.exit(1)
    
    print("\n✅ After installation, run training with:")
    print("    python finetune_conll2023.py")
    
    # Check if user wants to run the installation
    if len(sys.argv) > 1 and sys.argv[1] == '--install':
        print(f"\n🚀 Running: {install_cmd}")
        subprocess.run(install_cmd.split(), check=True)
        print("\n✅ Installation complete!")


if __name__ == "__main__":
    main()