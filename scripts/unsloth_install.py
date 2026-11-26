# noqa
# pylint: skip-file
try:
    import torch
except ImportError:
    raise ImportError("Install torch via `pip install torch`")
from packaging.version import Version as V

v = V(torch.__version__)
cuda = torch.version.cuda
try:
    is_ampere = torch.cuda.get_device_capability()[0] >= 8
except RuntimeError:
    is_ampere = False
# Accept common 11.x and 12.x CUDA versions and allow CPU-only (None).
if cuda is not None:
    cuda = str(cuda)
    if not (cuda.startswith("11.") or cuda.startswith("12.")):
        raise RuntimeError(f"CUDA = {cuda} not supported!")
else:
    # Keep cuda as None for CPU-only builds; handled when formatting the wheel tag.
    cuda = None
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
cuda_tag = cuda.replace(".", "") if cuda is not None else "cpu"
x = x.format(cuda_tag, "-ampere" if is_ampere else "")
print(
    f'pip install unsloth-zoo==2024.12.1 && pip install --no-deps "unsloth[{x}]==2024.12.4"'
)
