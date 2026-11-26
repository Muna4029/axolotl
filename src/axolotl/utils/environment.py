"""
utils to get GPU info for the current environment
"""

from accelerate.utils.environment import (
    check_cuda_p2p_ib_support as accelerate_check_cuda_p2p_ib_support,
    get_gpu_info,
)


def check_cuda_p2p_ib_support():
    if not accelerate_check_cuda_p2p_ib_support():
        return False
    unsupported_devices = frozenset(("RTX 6000 Ada", "L40S"))
    try:
        device_names, device_count = get_gpu_info()
        if 1 < device_count < 8:
            for device_name in device_names:
                for ud in unsupported_devices:
                    if ud in device_name:
                        return False
    except Exception:  # pylint: disable=broad-except # nosec
        pass
    return True
