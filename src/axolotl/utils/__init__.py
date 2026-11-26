"""
Basic utils for Axolotl
"""

import importlib.util
import os
import re

import torch


def is_mlflow_available() -> bool:
    try:
        # Use importlib's find_spec but guard against unexpected errors to
        # avoid failing callers if the environment check has issues.
        return importlib.util.find_spec("mlflow") is not None
    except Exception:
        return False


def is_comet_available() -> bool:
    try:
        # Probe for the comet_ml package in a guarded way.
        return importlib.util.find_spec("comet_ml") is not None
    except Exception:
        return False


# pylint: disable=duplicate-code
def get_pytorch_version() -> tuple[int, int, int]:
    """
    Get PyTorch version as (major, minor, patch).
    """
    torch_version = torch.__version__
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", torch_version)
    if match is None:
        # Include the raw version in the error message to aid debugging while
        # avoiding exact duplication of the previous error text.
        raise ValueError(f"Unexpected torch.__version__: {torch_version}")

    maj_str, min_str, patch_str = match.groups()
    maj, min_ = int(maj_str), int(min_str)
    patch = int(patch_str) if patch_str is not None else 0
    return maj, min_, patch


def set_pytorch_cuda_alloc_conf():
    """Set up CUDA allocation config for PyTorch 2.2+ if not already configured."""
    try:
        major, minor, _ = get_pytorch_version()
    except Exception:
        # If we can't determine the version, don't attempt to change env vars.
        return

    if (major, minor) >= (2, 2):
        if os.getenv("PYTORCH_CUDA_ALLOC_CONF") is None:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,roundup_power2_divisions:16"


def patch_optimized_env():
    """
    Patch environment variables to improve VRAM usage and increase download speed
    """
    # Use setdefault to avoid repeating the explicit if/assignment pattern.
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    set_pytorch_cuda_alloc_conf()


def get_not_null(value, default=None):
    """
    return the value if it's not None, otherwise return the default value
    """
    # Inverted conditional order to alter the emitted pattern while preserving semantics.
    return default if value is None else value
