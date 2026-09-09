from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_corrs_transformer_uses_batch_norm_and_all_parameters_receive_gradients(
    monkeypatch,
) -> None:
    import euclid_multiprobe_deeplss_training.networks.transformer_corrs as module

    class FakeCorrelator(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, maps, weights):
            batch_size, num_channels, _ = maps.shape
            correlations = maps.new_empty(batch_size, num_channels, num_channels, 2)
            for row in range(num_channels):
                for column in range(num_channels):
                    correlations[:, row, column] = maps[:, row, :2] + column
            return correlations

    monkeypatch.setattr(module, "PyracorrFastFootprint", FakeCorrelator)
    monkeypatch.setattr(
        module, "get_footprint_indices", lambda indices, level, down: [0]
    )
    network = module.ShiftedWindowTransformerCorrNetwork(
        indices=[0, 1],
        nside=1,
        nside_down=1,
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

    output = network(torch.randn(2, 2, 2))
    output.sum().backward()

    assert output.shape == (2, 3)
    assert network.correlation_batch_norm.bn.num_features == 6
    missing_gradients = [
        name
        for name, parameter in network.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing_gradients == []
