"""
test utils for helpers and decorators
"""

import os
from contextlib import contextmanager
from functools import wraps

from huggingface_hub.utils import reset_sessions


def reload_modules(hf_hub_offline):
    # Force reload of the modules that check this variable
    # Use __import__ to avoid top-level import statement inside the function
    importlib = __import__("importlib")
    datasets = __import__("datasets")
    huggingface_hub_constants = __import__("huggingface_hub.constants", fromlist=["constants"])

    # Reload the constants module first, as others depend on it
    importlib.reload(huggingface_hub_constants)
    huggingface_hub_constants.HF_HUB_OFFLINE = hf_hub_offline
    importlib.reload(datasets.config)
    setattr(datasets.config, "HF_HUB_OFFLINE", hf_hub_offline)
    reset_sessions()


def enable_hf_offline(test_func):
    """
    test decorator that sets HF_HUB_OFFLINE environment variable to True and restores it after the test even if the test fails.
    :param test_func:
    :return:
    """

    @wraps(test_func)
    def wrapper(*args, **kwargs):
        # Save the original value of HF_HUB_OFFLINE environment variable
        original_hf_offline = os.getenv("HF_HUB_OFFLINE")

        # Set HF_OFFLINE environment variable to True
        os.environ["HF_HUB_OFFLINE"] = "1"

        reload_modules(True)
        try:
            # Run the test function
            return test_func(*args, **kwargs)
        finally:
            # Restore the original value of HF_HUB_OFFLINE environment variable
            if original_hf_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = original_hf_offline
                # original_hf_offline is a string; compare to "1" to determine boolean state
                reload_modules(original_hf_offline == "1")
            else:
                del os.environ["HF_HUB_OFFLINE"]
                reload_modules(False)

    return wrapper


def disable_hf_offline(test_func):
    """
    test decorator that sets HF_HUB_OFFLINE environment variable to False and restores it after the wrapped func
    :param test_func:
    :return:
    """

    @wraps(test_func)
    def wrapper(*args, **kwargs):
        # Save the original value of HF_HUB_OFFLINE environment variable
        original_hf_offline = os.getenv("HF_HUB_OFFLINE")

        # Set HF_OFFLINE environment variable to True
        os.environ["HF_HUB_OFFLINE"] = "0"

        reload_modules(False)
        try:
            # Run the test function
            return test_func(*args, **kwargs)
        finally:
            # Restore the original value of HF_HUB_OFFLINE environment variable
            if original_hf_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = original_hf_offline
                # original_hf_offline is a string; compare to "1" to determine boolean state
                reload_modules(original_hf_offline == "1")
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
    reload_modules(bool(hf_hub_offline))
    yield
    # Restore the original value of HF_HUB_OFFLINE environment variable
    if original_hf_offline is not None:
        os.environ["HF_HUB_OFFLINE"] = original_hf_offline
        # original_hf_offline is a string; compare to "1" to determine boolean state
        reload_modules(original_hf_offline == "1")
    else:
        del os.environ["HF_HUB_OFFLINE"]
        reload_modules(False)
