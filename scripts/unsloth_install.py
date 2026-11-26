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
try:
    is_ampere = torch.cuda.get_device_capability()[0] >= 8
except RuntimeError:
    is_ampere = False
if cuda.lower() == "none":
    # CPU-only environment, allow
    pass
else:
    try:
        parts = cuda.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except Exception:
        # Unknown format — warn but continue to avoid hard failure in CI
        print(f"Warning: Unrecognized CUDA version format: {cuda}. Continuing...")
    else:
        # Allow 11.8 and any CUDA 12.x (including newer minor versions like 12.6)
        if not (major == 11 and minor == 8) and not (major >= 12):
            print(f"Warning: CUDA = {cuda} not officially supported, continuing...")
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
else:
    raise RuntimeError(f"Torch = {v} too new!")
x = x.format(cuda.replace(".", ""), "-ampere" if is_ampere else "")
uv_prefix = "uv " if use_uv else ""
print(
    f'{uv_prefix}pip install unsloth-zoo==2024.12.1 && {uv_prefix}pip install --no-deps "unsloth[{x}]==2024.12.4"'
)
