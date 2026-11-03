from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_corrs_transformer_uses_batch_norm_and_all_parameters_receive_gradients(
    monkeypatch,
) -> None:
    import euclid_multiprobe_deeplss_training.networks.transformer_corrs as module

    calls = {}

    class FakeCorrelator(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            calls["correlator_args"] = kwargs
            self.num_output_bins = 3

        def forward(self, maps, weights):
            batch_size, num_channels, _ = maps.shape
            correlations = maps.new_empty(batch_size, num_channels, num_channels, self.num_output_bins)
            for row in range(num_channels):
                for column in range(num_channels):
                    correlations[:, row, column] = maps[:, row, :3] + column
            return correlations

    monkeypatch.setattr(module, "NestcorrFastFootprint", FakeCorrelator)
    monkeypatch.setattr(
        module, "get_footprint_indices", lambda indices, level, down: [0]
    )
    network = module.ShiftedWindowTransformerCorrNetwork(
        indices=[0, 1, 2],
        nside=2,
        nside_down=2,
        num_channels=2,
        spins=[0, 0],
        embed_dim=3,
        weight_function=torch.ones_like,
        preprocess_function=lambda maps: maps,
        inner_embed_dim=4,
        depth=1,
        num_heads=1,
        window_size=2,
        dropout=0.0,
    )

    output = network(torch.randn(2, 3, 2))
    output.sum().backward()

    assert output.shape == (2, 3)
    assert network.correlation_batch_norm.bn.num_features == 9
    assert calls["correlator_args"]["pairs_filename"] == "pairs_L01_nd1.h5"
    assert calls["correlator_args"]["symmetrize"] is True
    assert calls["correlator_args"]["execution"] == "pytorch"
    missing_gradients = [
        name
        for name, parameter in network.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing_gradients == []


def test_corrs_transformer_builder_forwards_device_and_pair_cache(monkeypatch) -> None:
    from types import SimpleNamespace

    import euclid_multiprobe_deeplss_training.networks.transformer_corrs as module
    from euclid_multiprobe_deeplss_training.networks.builder import build_encoder

    calls = {}

    class FakeNetwork(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            calls.update(kwargs)

    monkeypatch.setattr(module, "ShiftedWindowTransformerCorrNetwork", FakeNetwork)
    physics = SimpleNamespace(
        channel_spins=[0],
        get_weight_maps=torch.ones_like,
        preprocess_for_correlations=lambda maps: maps,
    )
    model = build_encoder(
        "corrs_transformer", num_channels=1, embed_dim=3, nside=2, nside_down=1,
        num_pixels=4, indices=[0, 1, 2, 3], physics_model=physics,
        device="cpu", encoder_args={"pairs_filename": "angular_pairs.h5"},
    )

    assert isinstance(model, FakeNetwork)
    assert calls["device"] == "cpu"
    assert calls["pairs_filename"] == "angular_pairs.h5"
    assert calls["spins"] == [0]
