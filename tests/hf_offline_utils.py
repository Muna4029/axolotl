"""
test utils for helpers and decorators
"""

import os
from contextlib import contextmanager
from functools import wraps

from huggingface_hub.utils import reset_sessions


def reload_modules(hf_hub_offline):
    # Force reload of the modules that check this variable
    import importlib

    import datasets
    import huggingface_hub.constants

    # Reload the constants module first, as others depend on it
    importlib.reload(huggingface_hub.constants)
    huggingface_hub.constants.HF_HUB_OFFLINE = hf_hub_offline
    importlib.reload(datasets.config)
    setattr(datasets.config, "HF_HUB_OFFLINE", hf_hub_offline)
    reset_sessions()


def _set_hf_offline(value):
    """Set HF_HUB_OFFLINE env var to the given boolean value and reload modules.
    Returns the original environment value (or None if unset).
    """
    original = os.getenv("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1" if bool(value) else "0"
    reload_modules(bool(value))
    return original


def _restore_hf_offline(original):
    """Restore HF_HUB_OFFLINE from original and reload modules."""
    if original is not None:
        os.environ["HF_HUB_OFFLINE"] = original
        # Interpret the original value as boolean: recognize '1', 'true', 'yes' as truthy
        val = str(original).lower()
        reload_modules(val in ("1", "true", "yes"))
    else:
        # Remove the environment variable if it wasn't set originally
        if "HF_HUB_OFFLINE" in os.environ:
            del os.environ["HF_HUB_OFFLINE"]
        reload_modules(False)


def enable_hf_offline(test_func):
    """
    test decorator that sets HF_HUB_OFFLINE environment variable to True and restores it after the test even if the test fails.
    :param test_func:
    :return:
    """

    @wraps(test_func)
    def wrapper(*args, **kwargs):
        original = _set_hf_offline(True)
        try:
            return test_func(*args, **kwargs)
        finally:
            _restore_hf_offline(original)

    return wrapper


def disable_hf_offline(test_func):
    """
    test decorator that sets HF_HUB_OFFLINE environment variable to False and restores it after the wrapped func
    :param test_func:
    :return:
    """

    @wraps(test_func)
    def wrapper(*args, **kwargs):
        original = _set_hf_offline(False)
        try:
            return test_func(*args, **kwargs)
        finally:
            _restore_hf_offline(original)

    return wrapper


@contextmanager
def hf_offline_context(hf_hub_offline):
    """
    Context manager that sets HF_HUB_OFFLINE environment variable to the given value.
    :param hf_hub_offline: The new value for HF_HUB_OFFLINE.
    :return: A context manager.
    """
    original = os.getenv("HF_HUB_OFFLINE")
    # Use the helper to set the desired value (accepts truthy/falsy values)
    _set_hf_offline(hf_hub_offline)
    try:
        yield
    finally:
        _restore_hf_offline(original)
