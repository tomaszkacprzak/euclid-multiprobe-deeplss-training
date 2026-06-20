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


if "msfm" not in sys.modules:
    msfm_module = types.ModuleType("msfm")
    physics_pkg = types.ModuleType("msfm.onthefly_physics")
    physics_linear = types.ModuleType("msfm.onthefly_physics.onthefly_linear")
    pipeline_module = types.ModuleType("msfm.onthefly_pipeline")

    class _MissingOptionalDependency:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("msfm is not installed; tests should monkeypatch this dependency")

    physics_linear.OntheflyPhysicsModelLinear = _MissingOptionalDependency
    pipeline_module.OntheflyPipeline = _MissingOptionalDependency

    msfm_module.__spec__ = importlib.machinery.ModuleSpec("msfm", loader=None, is_package=True)
    physics_pkg.__spec__ = importlib.machinery.ModuleSpec("msfm.onthefly_physics", loader=None, is_package=True)
    physics_linear.__spec__ = importlib.machinery.ModuleSpec("msfm.onthefly_physics.onthefly_linear", loader=None)
    pipeline_module.__spec__ = importlib.machinery.ModuleSpec("msfm.onthefly_pipeline", loader=None)

    sys.modules["msfm"] = msfm_module
    sys.modules["msfm.onthefly_physics"] = physics_pkg
    sys.modules["msfm.onthefly_physics.onthefly_linear"] = physics_linear
    sys.modules["msfm.onthefly_pipeline"] = pipeline_module


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
