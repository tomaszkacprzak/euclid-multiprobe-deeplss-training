from __future__ import annotations

import sys
import types

import healpy as hp
import torch

from euclid_multiprobe_deeplss_training.networks.angular_power_spectra import AngularPowerSpectra


class _FakeTransform:
    def __init__(self, nside, *, lmax, mmax, quad_weights=None):
        self.lmax = lmax
        self.mmax = mmax

    def to(self, device):
        return self


class _FakeSHT(_FakeTransform):
    def __call__(self, signal):
        coeff = torch.zeros(self.lmax + 1, self.mmax + 1, dtype=torch.complex64, device=signal.device)
        total = signal.sum().to(torch.complex64)
        for ell in range(self.lmax + 1):
            coeff[ell, : min(ell, self.mmax) + 1] = total * (ell + 1)
        return coeff


class _FakeISHT(_FakeTransform):
    pass


def test_angular_power_spectra_shape(monkeypatch):
    fake_cuhpx = types.SimpleNamespace(SHTCUDA=_FakeSHT, iSHTCUDA=_FakeISHT)
    monkeypatch.setitem(sys.modules, "cuhpx", fake_cuhpx)

    nside = 1
    pixel_indices = torch.tensor([0, 3, 7, hp.nside2npix(nside) - 1])
    module = AngularPowerSpectra(nside=nside, pixel_file=pixel_indices, lmax=3, mmax=3)
    maps = torch.arange(2 * pixel_indices.numel() * 3, dtype=torch.float32).reshape(2, pixel_indices.numel(), 3)

    output = module(maps)

    assert output.shape == (2, 4, 6)
