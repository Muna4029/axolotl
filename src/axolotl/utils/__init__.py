"""
Basic utils for Axolotl
"""

import importlib.util
import os
import re

import torch


def is_mlflow_available() -> bool:
    return importlib.util.find_spec("mlflow") is not None


def is_comet_available() -> bool:
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

    # use .group(...) and assert non-None for groups that must be present
    major_str = version_match.group(1)
    minor_str = version_match.group(2)
    patch_str = version_match.group(3)

    assert major_str is not None and minor_str is not None
    major, minor = int(major_str), int(minor_str)
    patch = int(patch_str) if patch_str is not None else 0  # Default patch to 0 if not present
    return major, minor, patch


def set_pytorch_cuda_alloc_conf() -> None:
    """Set up CUDA allocation config if using PyTorch >= 2.2"""
    torch_version = torch.__version__.split(".")
    torch_major, torch_minor = int(torch_version[0]), int(torch_version[1])
    if torch_major == 2 and torch_minor >= 2:
        if os.getenv("PYTORCH_CUDA_ALLOC_CONF") is None:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                "expandable_segments:True,roundup_power2_divisions:16"
            )


def patch_optimized_env() -> None:
    """
    Patch environment variables to improve VRAM usage and increase download speed
    """
    if os.getenv("HF_HUB_ENABLE_HF_TRANSFER") is None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    set_pytorch_cuda_alloc_conf()
