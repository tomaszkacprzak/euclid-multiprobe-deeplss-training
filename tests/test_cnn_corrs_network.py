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
            self.num_output_bins = 3

        def forward(self, maps, weights):
            calls["maps"] = maps.clone()
            calls["weights"] = weights.clone()
            batch_size, num_channels, _ = maps.shape
            values = maps.new_zeros(batch_size, num_channels, num_channels, self.num_output_bins)
            for row in range(num_channels):
                for column in range(num_channels):
                    values[:, row, column] = 10 * row + column
            return values

    monkeypatch.setattr(module, "NestcorrFastFootprint", FakeCorrelator)
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
        pairs_filename="angular_pairs.h5",
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
    assert network.num_corrs == 3
    assert network.correlation_batch_norm.bn.num_features == 9
    assert calls["correlator_args"]["pairs_filename"] == "angular_pairs.h5"
    assert calls["correlator_args"]["symmetrize"] is True
    assert calls["correlator_args"]["execution"] == "pytorch"
    assert "doublesets" not in calls["correlator_args"]
    assert "recompute_pairs" not in calls["correlator_args"]
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


@pytest.mark.parametrize("binning", ["standard", "angular"])
@pytest.mark.parametrize("encoder", ["cnn", "transformer"])
@pytest.mark.parametrize("masked_channel", [False, True])
def test_correlation_encoders_use_cached_nestcorr_bins_and_backpropagate(tmp_path, binning, encoder, masked_channel) -> None:
    from nestcorr.generate_pairs import generate_pairs

    from euclid_multiprobe_deeplss_training.networks.cnn_corrs import ConvolutionalResidualCorrNetwork
    from euclid_multiprobe_deeplss_training.networks.transformer_corrs import ShiftedWindowTransformerCorrNetwork

    pairs = generate_pairs(2, tmp_path / "pairs.h5", binning=binning)

    def weights(maps):
        return torch.ones_like(maps.real) * maps.real.new_tensor([1, 0 if masked_channel else 1])

    arguments = dict(
        indices=[0, 1, 2, 3],
        nside=4,
        nside_down=2,
        num_channels=2,
        spins=[0, 2],
        embed_dim=3,
        pairs_filename=str(pairs),
        weight_function=weights,
        preprocess_function=lambda maps: maps,
        dropout=0.0,
    )
    if encoder == "cnn":
        network = ConvolutionalResidualCorrNetwork(
            **arguments, inner_channels=4, downsampling_layers=1, residual_layers=1,
        )
    else:
        network = ShiftedWindowTransformerCorrNetwork(
            **arguments, inner_embed_dim=4, depth=1, num_heads=1, window_size=2,
        )

    maps = torch.complex(torch.randn(3, 4, 2), torch.randn(3, 4, 2))
    maps[..., 0] = maps[..., 0].real.to(torch.complex64)
    maps.requires_grad_()
    normalization_inputs = []
    network.correlation_batch_norm.register_forward_pre_hook(
        lambda _module, inputs: normalization_inputs.append(inputs[0].detach().clone())
    )

    output = network(maps)
    output.square().sum().backward()

    assert output.shape == (3, 3)
    assert torch.isfinite(output).all()
    assert maps.grad is not None and torch.isfinite(maps.grad).all()
    assert maps.grad.abs().sum() > 0
    if masked_channel:
        assert torch.count_nonzero(maps.grad[..., 1]) == 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in network.parameters())
    expected_bins = 3 if binning == "standard" else network.correlator.angular_bin_edges.numel() - 1
    assert network.num_corrs == expected_bins
    assert network.correlation_batch_norm.bn.num_features == 3 * expected_bins
    correlations = network.correlator(maps.detach().movedim(-1, 1), weights(maps).movedim(-1, 1))
    assert torch.isnan(correlations).any(), "the small footprint must exercise empty-bin handling"
    upper = torch.triu_indices(2, 2)
    expected = torch.nan_to_num(correlations, nan=0.0)[:, upper[0], upper[1], :].transpose(1, 2)
    torch.testing.assert_close(normalization_inputs[0], expected)
