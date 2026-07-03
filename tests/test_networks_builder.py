from __future__ import annotations

import importlib
import sys
import types

import pytest


def test_build_model_passes_model_args_as_constructor_kwargs(monkeypatch) -> None:
    pytest.importorskip("torch")

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


def test_deepsphere_resnet_head_is_not_lazy(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")

    fake_deepsphere = types.ModuleType("deepsphere")
    fake_layers = types.SimpleNamespace()

    class FakeHealpyGCNN(torch.nn.Module):
        def __init__(self, *, layers, **kwargs):
            super().__init__()
            self.layers_use = torch.nn.ModuleList(layers)
            self.kwargs = kwargs

    class FakeGraphLayer(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs

    class FakeResidualLayer(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.args = args
            self.kwargs = kwargs

    fake_layers.HealpyPseudoConv = FakeGraphLayer
    fake_layers.HealpyChebyshev = FakeGraphLayer
    fake_layers.Healpy_ResidualLayer = FakeResidualLayer
    fake_deepsphere.HealpyGCNN = FakeHealpyGCNN
    fake_deepsphere.healpy_layers = fake_layers

    fake_utils = types.ModuleType("deepsphere.utils")
    fake_utils.extend_indices = lambda indices, **_kwargs: np.asarray(indices)

    monkeypatch.setitem(sys.modules, "deepsphere", fake_deepsphere)
    monkeypatch.setitem(sys.modules, "deepsphere.utils", fake_utils)
    sys.modules.pop("euclid_multiprobe_deeplss_training.networks.deepsphere_resnet", None)

    deepsphere_resnet = importlib.import_module("euclid_multiprobe_deeplss_training.networks.deepsphere_resnet")
    model = deepsphere_resnet.ResnetDeepSphereRegressor(
        n_side=32,
        indices=list(range(12 * 32 * 32)),
        batch_size=2,
        n_channels=1,
        out_features=3,
        n_filters=16,
        downsampling_layers=3,
        cheby_layers=2,
        residual_layers=4,
        poly_degree=5,
        n_neighbors=20,
    )

    assert not any(isinstance(module, torch.nn.modules.lazy.LazyModuleMixin) for module in model.modules())
    assert isinstance(model.layers_use[-2], torch.nn.LayerNorm)
    assert isinstance(model.layers_use[-1], torch.nn.Linear)
    assert model.layers_use[-2].normalized_shape == (1536,)
    assert model.layers_use[-1].in_features == 1536
