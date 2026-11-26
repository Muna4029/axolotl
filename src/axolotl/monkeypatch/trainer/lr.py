"""
monkeypatch for Trainer _get_learning_rate method
"""

# Lazy-import heavy or project modules to avoid importing them at module import time
# This prevents pre-commit / static-analysis hook environments from trying to
# install or import large/unavailable dependencies during initialization.
class _LazyModule:
    def __init__(self, import_name):
        self._import_name = import_name
        self._mod = None

    def _load(self):
        if self._mod is None:
            try:
                import importlib

                self._mod = importlib.import_module(self._import_name)
            except Exception:
                self._mod = None
        return self._mod

    def __getattr__(self, name):
        mod = self._load()
        if mod is None:
            raise RuntimeError(f"Module {self._import_name} is not available")
        return getattr(mod, name)

# Expose a lazy 'torch' proxy so other code can reference 'torch' without
# triggering an immediate import during static analysis / hook setup.
torch = _LazyModule("torch")

# Provide a lazy get_logger function that imports the real project logger when
# it's actually called. Fall back to the standard library logging when the
# project logger cannot be imported (e.g., in lightweight hook environments).

def get_logger(*args, **kwargs):
    try:
        from axolotl.utils.logging import get_logger as _real_get_logger

        return _real_get_logger(*args, **kwargs)
    except Exception:
        import logging

        return logging.getLogger(args[0] if args else __name__)

LOG = get_logger(__name__)


# TODO remove this patch once https://github.com/huggingface/transformers/pull/37881 is included in a release
def _get_learning_rate(self):
    if self.is_deepspeed_enabled:
        # with deepspeed's fp16 and dynamic loss scale enabled the optimizer/scheduler steps may
        # not run for the first few dozen steps while loss scale is too large, and thus during
        # that time `get_last_lr` will fail if called during that warm up stage, so work around it:
        try:
            last_lr = self.lr_scheduler.get_last_lr()[0]
        except AssertionError as e:
            if "need to call step" in str(e):
                LOG.warning(
                    "tried to get lr value before scheduler/optimizer started stepping, returning lr=0"
                )
                last_lr = 0
            else:
                raise
    else:
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            last_lr = self.optimizer.param_groups[0]["lr"]
        else:
            last_lr = self.lr_scheduler.get_last_lr()[0]

    if torch.is_tensor(last_lr):
        last_lr = last_lr.item()
    return last_lr


def patch_trainer_get_lr():
    from transformers.trainer import Trainer

    Trainer._get_learning_rate = _get_learning_rate  # pylint: disable=protected-access
