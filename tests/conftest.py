from __future__ import annotations

import importlib.machinery
import sys
import types

if "tqdm" not in sys.modules:
    tqdm_module = types.ModuleType("tqdm")

    def _tqdm(collection, **_kwargs):
        return collection

    tqdm_module.tqdm = _tqdm
    tqdm_module.__spec__ = importlib.machinery.ModuleSpec("tqdm", loader=None)
    sys.modules["tqdm"] = tqdm_module


if "wandb" not in sys.modules:
    wandb_module = types.ModuleType("wandb")

    def _wandb_init(**_kwargs):
        return None

    def _wandb_log(*_args, **_kwargs):
        return None

    wandb_module.init = _wandb_init
    wandb_module.log = _wandb_log
    wandb_module.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
    sys.modules["wandb"] = wandb_module


if "psutil" not in sys.modules:
    psutil_module = types.ModuleType("psutil")

    class _MemoryInfo:
        rss = 0

    class _Process:
        def __init__(self, *_args, **_kwargs):
            pass

        def memory_info(self):
            return _MemoryInfo()

    psutil_module.Process = _Process
    psutil_module.__spec__ = importlib.machinery.ModuleSpec("psutil", loader=None)
    sys.modules["psutil"] = psutil_module
