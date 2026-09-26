# noqa
# pylint: skip-file
import sys

try:
    import torch
except ImportError:
    raise ImportError("Install torch via `pip install torch`")
from packaging.version import Version as V

use_uv = "--uv" in sys.argv[1:]

v = V(torch.__version__)
cuda = str(torch.version.cuda)

# Handle case where CUDA is not available (CPU-only torch)
if cuda == "None" or cuda is None:
    print("")
    sys.exit(0)

try:
    is_ampere = torch.cuda.get_device_capability()[0] >= 8
except (RuntimeError, AssertionError):
    # CUDA not available or not compiled with CUDA
    is_ampere = False

# Check if CUDA version is supported
supported_cuda = ["12.1", "11.8", "12.4", "12.6", "12.8", "13.0"]
if cuda not in supported_cuda:
    raise RuntimeError(f"CUDA = {cuda} not supported!")

if v <= V("2.1.0"):
    raise RuntimeError(f"Torch = {v} too old!")
elif v <= V("2.1.1"):
    x = "cu{}{}-torch211"
elif v <= V("2.1.2"):
    x = "cu{}{}-torch212"
elif v < V("2.3.0"):
    x = "cu{}{}-torch220"
elif v < V("2.4.0"):
    x = "cu{}{}-torch230"
elif v < V("2.5.0"):
    x = "cu{}{}-torch240"
elif v < V("2.6.0"):
    x = "cu{}{}-torch250"
elif v < V("2.7.0"):
    x = "cu{}{}-torch260"
elif v < V("2.8.0"):
    x = "cu{}{}-torch270"
elif v < V("2.9.0"):
    x = "cu{}{}-torch280"
elif v < V("2.10.0"):
    x = "cu{}{}-torch290"
elif v < V("2.11.0"):
    x = "cu{}{}-torch2100"
elif v < V("2.12.0"):
    x = "cu{}{}-torch2110"
elif v < V("2.13.0"):
    x = "cu{}{}-torch2120"
else:
    x = "cu{}{}-torch2130"

x = x.format(cuda.replace(".", ""), "-ampere" if is_ampere else "")
uv_prefix = "uv " if use_uv else ""
print(
    f'{uv_prefix}pip install unsloth-zoo==2024.12.1 && {uv_prefix}pip install --no-deps "unsloth[{x}]==2024.12.4"'
)
