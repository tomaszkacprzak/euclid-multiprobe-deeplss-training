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
    fake_cuhpx.nest2ring = lambda maps, nside: maps.flip(-1)
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
            (
                alms[:, 2, 0].abs().square()
                + 2 * alms[:, 2, 1].abs().square()
                + 2 * alms[:, 2, 2].abs().square()
            )
            / 5,
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
