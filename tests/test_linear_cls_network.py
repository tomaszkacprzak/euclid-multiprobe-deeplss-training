from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_linear_cls_network_projects_all_auto_and_cross_spectra(monkeypatch) -> None:
    import euclid_multiprobe_deeplss_training.networks.linear_cls as module

    class FakePartSkyCls(torch.nn.Module):
        def __init__(self, indices, nside, lmax, sub_batch_size):
            super().__init__()
            self.lmax = lmax

        def forward(self, *maps):
            batch_size = maps[0].shape[0]
            num_spectra = len(maps) * (len(maps) + 1) // 2
            return maps[0].new_ones(batch_size, self.lmax, num_spectra)

    monkeypatch.setattr(module, "PartSkyCls", FakePartSkyCls)
    network = module.LinearClsNetwork(indices=[0, 1, 2], nside=1, num_channels=3, embed_dim=5, lmax=4)

    output = network(torch.randn(2, 3, 3))

    assert module.LinearClsNetwork.tag == "cls_linear"
    assert network.linear.in_features == 4 * 6
    assert output.shape == (2, 5)


def test_linear_cls_network_average_pools_along_ell(monkeypatch) -> None:
    import euclid_multiprobe_deeplss_training.networks.linear_cls as module

    class FakePartSkyCls(torch.nn.Module):
        def __init__(self, indices, nside, lmax, sub_batch_size):
            super().__init__()
            self.lmax = lmax

        def forward(self, *maps):
            ell = torch.arange(self.lmax, dtype=maps[0].dtype, device=maps[0].device)
            return ell[None, :, None].expand(maps[0].shape[0], -1, len(maps))

    monkeypatch.setattr(module, "PartSkyCls", FakePartSkyCls)
    network = module.LinearClsNetwork(
        indices=[0, 1, 2],
        nside=2,
        num_channels=1,
        embed_dim=2,
        lmax=6,
        window_size=2,
        unstack_function=lambda maps: (maps[..., 0],),
    )
    captured = {}
    network.linear.register_forward_pre_hook(lambda _module, inputs: captured.setdefault("input", inputs[0]))

    output = network(torch.randn(1, 3, 1))

    assert output.shape == (1, 2)
    assert network.linear.in_features == 3
    assert network.downsample is not None
    assert network.downsample.stride == (2,)
    assert not network.downsample.weight.requires_grad
    torch.testing.assert_close(captured["input"], torch.tensor([[0.5, 2.5, 4.5]]))


@pytest.mark.parametrize("window_size", [0, -1, 7])
def test_linear_cls_network_rejects_invalid_window_size(monkeypatch, window_size) -> None:
    import euclid_multiprobe_deeplss_training.networks.linear_cls as module

    monkeypatch.setattr(module, "PartSkyCls", lambda *args, **kwargs: torch.nn.Identity())

    with pytest.raises(ValueError, match="window_size"):
        module.LinearClsNetwork(
            indices=[0], nside=2, num_channels=1, embed_dim=2, lmax=6, window_size=window_size
        )


def test_builder_creates_cls_linear_encoder(monkeypatch) -> None:
    import euclid_multiprobe_deeplss_training.networks.linear_cls as module
    from euclid_multiprobe_deeplss_training.networks.builder import build_encoder

    captured = {}

    class FakeLinearClsNetwork:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(module, "LinearClsNetwork", FakeLinearClsNetwork)
    encoder = build_encoder(
        "cls_linear",
        num_channels=2,
        embed_dim=7,
        nside=16,
        nside_down=4,
        num_pixels=12,
        indices=[2, 4, 6],
        encoder_args={"lmax": 20, "sub_batch_size": 2},
    )

    assert isinstance(encoder, FakeLinearClsNetwork)
    assert captured == {
        "indices": [2, 4, 6],
        "nside": 16,
        "num_channels": 2,
        "embed_dim": 7,
        "lmax": 20,
        "sub_batch_size": 2,
    }
