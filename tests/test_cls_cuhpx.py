from __future__ import annotations

import importlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")


class _FakeSHT:
    def __init__(self, **kwargs: object) -> None:
        self.lmax = int(kwargs["lmax"])
        self.mmax = int(kwargs["mmax"])

    def __call__(self, maps: torch.Tensor) -> torch.Tensor:
        batch_size = maps.shape[0]
        values = torch.arange(
            batch_size * self.lmax * self.mmax,
            device=maps.device,
            dtype=maps.dtype,
        )
        return values.reshape(batch_size, self.lmax, self.mmax).to(torch.complex64)


@pytest.fixture
def cls_module(monkeypatch: pytest.MonkeyPatch):
    fake_cuhpx = types.ModuleType("cuhpx")
    fake_cuhpx.SHTCUDA = _FakeSHT
    fake_cuhpx.iSHTCUDA = _FakeSHT
    fake_cuhpx.nest2ring = lambda maps, nside: maps.flip(-1)
    fake_cuhpx.ring2nest = lambda maps, nside: maps.flip(-1)
    monkeypatch.setitem(sys.modules, "cuhpx", fake_cuhpx)
    sys.modules.pop("euclid_multiprobe_deeplss_training.utils.cls_cuhpx", None)
    return importlib.import_module("euclid_multiprobe_deeplss_training.utils.cls_cuhpx")


def test_cls_weights_nonnegative_m_modes(cls_module, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    transform = cls_module.AutoClsCuHPX(nside=1, lmax=3)
    maps = torch.zeros(2, 12)

    result = transform(maps)

    alms = transform.sht(maps)
    expected = torch.stack(
        [
            alms[:, 0, 0].abs().square(),
            (alms[:, 1, 0].abs().square() + 2 * alms[:, 1, 1].abs().square()) / 3,
            (alms[:, 2, 0].abs().square() + 2 * alms[:, 2, 1].abs().square() + 2 * alms[:, 2, 2].abs().square()) / 5,
        ],
        dim=1,
    )
    torch.testing.assert_close(result, expected)
    assert result.shape == (2, 3)


def test_cls_validates_map_shape_and_full_m_range(cls_module) -> None:
    with pytest.raises(ValueError, match="mmax must equal lmax"):
        cls_module.AutoClsCuHPX(nside=2, lmax=4, mmax=3)

    transform = cls_module.AutoClsCuHPX(nside=2, lmax=4)
    with pytest.raises(ValueError, match="Expected 48 pixels"):
        transform(torch.zeros(1, 47))


def test_spin2_cls_returns_e_and_b_auto_spectra(cls_module, monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeEB:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["lmax"] == 2
            assert kwargs["output_type"] == "alm"

        def __call__(self, maps: torch.Tensor) -> torch.Tensor:
            batch_size = maps.shape[0]
            values = torch.arange(
                batch_size * 2 * 3 * 3,
                device=maps.device,
                dtype=maps.dtype,
            )
            return values.reshape(batch_size, 2, 3, 3).to(torch.complex64)

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(cls_module, "CuHPXScalarRouteEB", _FakeEB)
    transform = cls_module.AutoClsCuHPX(nside=1, lmax=3)
    maps = torch.zeros(2, 2, 12)

    result = transform(maps)

    alms = transform._spin_sht(maps.flip(-1))
    expected = torch.stack(
        (
            alms[..., 0, 0].abs().square(),
            (alms[..., 1, 0].abs().square() + 2 * alms[..., 1, 1].abs().square()) / 3,
            (alms[..., 2, 0].abs().square() + 2 * alms[..., 2, 1].abs().square() + 2 * alms[..., 2, 2].abs().square()) / 5,
        ),
        dim=-1,
    )
    torch.testing.assert_close(result, expected)
    assert result.shape == (2, 2, 3)


def test_spin2_cls_validates_channels_and_lmax(cls_module, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    transform = cls_module.AutoClsCuHPX(nside=1, lmax=2)

    with pytest.raises(ValueError, match="exactly two channels"):
        transform(torch.zeros(1, 3, 12))
    with pytest.raises(ValueError, match="at least 3"):
        transform(torch.zeros(1, 2, 12))


def test_part_sky_cls_expands_and_processes_one_batch_item(cls_module, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class _FakeAutoCls:
        num_pixels = 12
        lmax = 3

        def __init__(self, **kwargs: object) -> None:
            assert kwargs["nside"] == 1
            assert kwargs["lmax"] == 3
            assert kwargs["input_order"] == "nest"

        def __call__(self, maps: torch.Tensor) -> torch.Tensor:
            calls.append(maps.clone())
            return maps.sum(dim=-1, keepdim=True).expand(*maps.shape[:-1], self.lmax)

    monkeypatch.setattr(cls_module, "AutoClsCuHPX", _FakeAutoCls)
    transform = cls_module.PartSkyAutoCls(torch.tensor([1, 4, 8]), nside=1, lmax=3)
    scalar = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    spin2 = torch.arange(6.0).reshape(1, 2, 3)

    scalar_cls, spin2_cls = transform(scalar, spin2)

    assert scalar_cls.shape == (2, 3)
    assert spin2_cls.shape == (1, 2, 3)
    assert len(calls) == 3
    assert all(call.shape[0] == 1 for call in calls)
    torch.testing.assert_close(calls[0][0, [1, 4, 8]], scalar[0])
    assert torch.count_nonzero(calls[0]) == 3


def test_part_sky_cls_validates_indices_and_map_shape(cls_module, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        cls_module.PartSkyAutoCls(torch.tensor([1, 1]), nside=1, lmax=3)
    with pytest.raises(ValueError, match="range"):
        cls_module.PartSkyAutoCls(torch.tensor([12]), nside=1, lmax=3)

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    transform = cls_module.PartSkyAutoCls(torch.tensor([1, 4]), nside=1, lmax=3)
    with pytest.raises(ValueError, match="Expected 2 part-sky pixels"):
        transform(torch.zeros(1, 3))
