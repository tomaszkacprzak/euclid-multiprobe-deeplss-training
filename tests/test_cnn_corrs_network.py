from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_cnn_corrs_preprocesses_maps_and_convolves_unique_correlations(
    monkeypatch,
) -> None:
    import euclid_multiprobe_deeplss_training.networks.cnn_corrs as module

    calls = {}

    class FakeCorrelator(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            calls["correlator_args"] = kwargs

        def forward(self, maps, weights):
            calls["maps"] = maps.clone()
            calls["weights"] = weights.clone()
            batch_size, num_channels, _ = maps.shape
            values = maps.new_zeros(batch_size, num_channels, num_channels, 2)
            for row in range(num_channels):
                for column in range(num_channels):
                    values[:, row, column] = 10 * row + column
            return values

    monkeypatch.setattr(module, "PyracorrFastFootprint", FakeCorrelator)
    monkeypatch.setattr(
        module, "get_footprint_indices", lambda indices, level, down: [0]
    )

    def weights(maps):
        calls["weight_input"] = maps.clone()
        return maps + 2

    def preprocess(maps):
        calls["preprocess_input"] = maps.clone()
        return maps - 1

    network = module.ConvolutionalResidualCorrNetwork(
        indices=[0, 1, 2],
        nside=1,
        nside_down=1,
        num_channels=2,
        spins=[0, 2],
        embed_dim=5,
        weight_function=weights,
        preprocess_function=preprocess,
        inner_channels=8,
        downsampling_layers=1,
        residual_layers=1,
        dropout=0.0,
    )
    input_maps = torch.randn(2, 3, 2)

    output = network(input_maps)

    assert network.tag == "corr_cnn"
    assert network.downsampling[0].in_channels == 3
    assert output.shape == (2, 5)
    torch.testing.assert_close(calls["weight_input"], input_maps)
    torch.testing.assert_close(calls["preprocess_input"], input_maps)
    torch.testing.assert_close(
        calls["maps"], torch.movedim(input_maps - 1, -1, 1)
    )
    torch.testing.assert_close(
        calls["weights"], torch.movedim(input_maps + 2, -1, 1)
    )


def test_cnn_corrs_validates_spin_count(monkeypatch) -> None:
    import euclid_multiprobe_deeplss_training.networks.cnn_corrs as module

    with pytest.raises(ValueError, match="spins must have the same length"):
        module.ConvolutionalResidualCorrNetwork(
            indices=[0],
            nside=1,
            nside_down=1,
            num_channels=2,
            spins=[0],
            embed_dim=2,
        )
