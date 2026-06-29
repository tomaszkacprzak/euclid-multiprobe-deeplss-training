from __future__ import annotations

import sys
import types


def test_build_model_passes_model_args_as_constructor_kwargs(monkeypatch) -> None:
    from euclid_multiprobe_deeplss_training.networks.builder import build_model

    captured = {}
    fake_module = types.ModuleType("euclid_multiprobe_deeplss_training.networks.healpix_transformer")

    class FakeTransformer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module.HealpixNestedHierarchicalLocalWindowTransformer = FakeTransformer
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    model = build_model(
        "nested_transformer",
        num_channels=3,
        num_targets=2,
        nside=512,
        nside_down=64,
        num_pixels=1024,
        model_args={"base_embed_dim": 512, "growth": "256", "num_heads": 8},
    )

    assert isinstance(model, FakeTransformer)
    assert captured == {
        "nside": 512,
        "nside_down": 64,
        "num_pixels": 1024,
        "in_channels": 3,
        "num_outputs": 2,
        "base_embed_dim": 512,
        "growth": "256",
        "num_heads": 8,
        "window_levels": 3,
        "local_blocks_per_level": 1,
        "global_blocks": 1,
        "mlp_ratio": 4,
    }
