"""
Basic utils for Axolotl
"""

import importlib.util
import os
import re

# Avoid importing heavy optional dependency `torch` at module import time.
# Import torch lazily inside functions that need it to prevent pre-commit
# environments from attempting to install it.


def is_mlflow_available():
    return importlib.util.find_spec("mlflow") is not None


def is_comet_available():
    return importlib.util.find_spec("comet_ml") is not None


# pylint: disable=duplicate-code
def get_pytorch_version() -> tuple[int, int, int]:
    """
    Get Pytorch version as a tuple of (major, minor, patch).
    """
    torch_version = torch.__version__
    version_match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", torch_version)

    if not version_match:
        raise ValueError("Invalid version format")

    major, minor, patch = version_match.groups()
    major, minor = int(major), int(minor)
    patch = int(patch) if patch is not None else 0  # Default patch to 0 if not present
    return major, minor, patch


def set_pytorch_cuda_alloc_conf():
    """Set up CUDA allocation config if using PyTorch >= 2.2

    This function performs a lazy import of torch and uses a defensive
    version parse to avoid ValueError on non-numeric version components
    (e.g. '2.2rc1' or '2.2.0+cu118'). If torch is not available, the
    function is a no-op.
    """
    try:
        import torch  # imported lazily to avoid top-level dependency
    except Exception:
        return

    torch_ver = getattr(torch, "__version__", "")
    # Match major and minor numeric components at the start of the version
    m = re.match(r"^(\d+)\.(\d+)", str(torch_ver))
    if not m:
        return

    try:
        torch_major = int(m.group(1))
        torch_minor = int(m.group(2))
    except ValueError:
        return

    if torch_major == 2 and torch_minor >= 2:
        if os.getenv("PYTORCH_CUDA_ALLOC_CONF") is None:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                "expandable_segments:True,roundup_power2_divisions:16"
            )


def patch_optimized_env():
    """
    Patch environment variables to improve VRAM usage and increase download speed
    """
    if os.getenv("HF_HUB_ENABLE_HF_TRANSFER") is None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    set_pytorch_cuda_alloc_conf()


def get_not_null(value, default=None):
    """
    return the value if it's not None, otherwise return the default value
    """
    return value if value is not None else default
