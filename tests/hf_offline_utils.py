"""
test utils for helpers and decorators
"""

import os
from contextlib import contextmanager
from functools import wraps

try:
    from huggingface_hub.utils import reset_sessions
except Exception:  # pragma: no cover - provide a noop fallback when huggingface_hub is not installed in CI
    def reset_sessions():
        return None


def reload_modules(hf_hub_offline):
    # Force reload of the modules that check this variable
    import importlib

    import datasets
    import huggingface_hub.constants

    # Normalize the value to a consistent string representation "1" or "0"
    if isinstance(hf_hub_offline, str):
        hf_hub_offline_str = hf_hub_offline
    else:
        hf_hub_offline_str = "1" if bool(hf_hub_offline) else "0"

    # Reload the constants module first, as others depend on it
    importlib.reload(huggingface_hub.constants)
    huggingface_hub.constants.HF_HUB_OFFLINE = hf_hub_offline_str
    importlib.reload(datasets.config)
    setattr(datasets.config, "HF_HUB_OFFLINE", hf_hub_offline_str)
    reset_sessions()


def enable_hf_offline(test_func):
    """
    test decorator that sets HF_HUB_OFFLINE environment variable to '1' and restores it after the test even if the test fails.
    :param test_func:
    :return:
    """

    @wraps(test_func)
    def wrapper(*args, **kwargs):
        # Save the original value of HF_HUB_OFFLINE environment variable
        original_hf_offline = os.getenv("HF_HUB_OFFLINE")

        # Set HF_HUB_OFFLINE environment variable to '1'
        os.environ["HF_HUB_OFFLINE"] = "1"

        reload_modules(True)
        try:
            # Run the test function
            return test_func(*args, **kwargs)
        finally:
            # Restore the original value of HF_HUB_OFFLINE environment variable
            if original_hf_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = original_hf_offline
                reload_modules(original_hf_offline)
            else:
                del os.environ["HF_HUB_OFFLINE"]
                reload_modules(False)

    return wrapper


def disable_hf_offline(test_func):
    """
    test decorator that sets HF_HUB_OFFLINE environment variable to '0' and restores it after the wrapped func
    :param test_func:
    :return:
    """

    @wraps(test_func)
    def wrapper(*args, **kwargs):
        # Save the original value of HF_HUB_OFFLINE environment variable
        original_hf_offline = os.getenv("HF_HUB_OFFLINE")

        # Set HF_HUB_OFFLINE environment variable to '0'
        os.environ["HF_HUB_OFFLINE"] = "0"

        reload_modules(False)
        try:
            # Run the test function
            return test_func(*args, **kwargs)
        finally:
            # Restore the original value of HF_HUB_OFFLINE environment variable
            if original_hf_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = original_hf_offline
                reload_modules(original_hf_offline)
            else:
                del os.environ["HF_HUB_OFFLINE"]
                reload_modules(False)

    return wrapper


@contextmanager
def hf_offline_context(hf_hub_offline):
    """
    Context manager that sets HF_HUB_OFFLINE environment variable to the given value.
    :param hf_hub_offline: The new value for HF_HUB_OFFLINE.
    :return: A context manager.
    """
    original_hf_offline = os.getenv("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = str(hf_hub_offline)
    reload_modules(hf_hub_offline)
    try:
        yield
    finally:
        # Restore the original value of HF_HUB_OFFLINE environment variable
        if original_hf_offline is not None:
            os.environ["HF_HUB_OFFLINE"] = original_hf_offline
            reload_modules(original_hf_offline)
        else:
            del os.environ["HF_HUB_OFFLINE"]
            reload_modules(False)
