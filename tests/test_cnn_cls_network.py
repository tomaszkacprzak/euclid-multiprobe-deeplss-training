from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_cnn_cls_network_passes_all_spectra_to_convolutions(monkeypatch) -> None:
    import euclid_multiprobe_deeplss_training.networks.cnn_cls as module

    class FakePartSkyCls(torch.nn.Module):
        def __init__(self, indices, nside, lmax, sub_batch_size):
            super().__init__()
            self.lmax = lmax

        def forward(self, *maps):
            num_spectra = len(maps) * (len(maps) + 1) // 2
            return maps[0].new_ones(maps[0].shape[0], self.lmax, num_spectra)

    monkeypatch.setattr(module, "PartSkyCls", FakePartSkyCls)
    network = module.ConvolutionalResidualClsNetwork(
        indices=[0, 1, 2],
        nside=1,
        num_channels=3,
        embed_dim=5,
        lmax=4,
        unstack_function=lambda maps: maps.unbind(dim=-1),
        inner_channels=8,
        downsampling_layers=2,
        residual_layers=2,
        dropout=0.0,
    )

    output = network(torch.randn(2, 3, 3))

    assert network.tag == "cls_cnn"
    assert network.downsampling[0].in_channels == 6
    assert output.shape == (2, 5)


def test_cnn_cls_network_validates_map_channels(monkeypatch) -> None:
    import euclid_multiprobe_deeplss_training.networks.cnn_cls as module

    monkeypatch.setattr(module, "PartSkyCls", lambda *args, **kwargs: torch.nn.Identity())
    network = module.ConvolutionalResidualClsNetwork(
        indices=[0],
        nside=1,
        num_channels=1,
        embed_dim=2,
        unstack_function=lambda maps: maps.unbind(dim=-1),
    )

    with pytest.raises(ValueError, match="Expected 1 input channels"):
        network(torch.randn(2, 3, 2))
