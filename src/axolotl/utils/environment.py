"""
utils to get GPU info for the current environment
"""

from typing import Any, Callable

# Defer importing accelerate until the functions are actually called to avoid
# importing heavy/optional dependencies at module import time (which can
# interfere with pre-commit / CI environment initialization).
def _lazy_accelerate_func(attr: str) -> Callable[..., Any]:
    def _caller(*args, **kwargs):
        mod = __import__("accelerate.utils.environment", fromlist=[attr])
        return getattr(mod, attr)(*args, **kwargs)
    return _caller

accelerate_check_cuda_p2p_ib_support = _lazy_accelerate_func("check_cuda_p2p_ib_support")
get_gpu_info = _lazy_accelerate_func("get_gpu_info")


def check_cuda_p2p_ib_support():
    if not accelerate_check_cuda_p2p_ib_support():
        return False
    unsupported_devices = {"RTX 6000 Ada", "L40S"}
    try:
        device_names, device_count = get_gpu_info()
        if 1 < device_count < 8:
            if any(
                unsupported_device in device_name
                for device_name in device_names
                for unsupported_device in unsupported_devices
            ):
                return False
    except Exception:  # pylint: disable=broad-except # nosec
        pass
    return True
