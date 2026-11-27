from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from euclid_multiprobe_deeplss_training.networks.mse_loss import MSEModel
from euclid_multiprobe_deeplss_training.training import evaluate


def _identity_model(encoder=None):
    model = MSEModel(torch.nn.Identity() if encoder is None else encoder, embed_dim=2, num_targets=2)
    with torch.no_grad():
        model.loss.linear.weight.copy_(torch.eye(2))
        model.loss.linear.bias.zero_()
    return model


@pytest.mark.parametrize("channel_list", [False, True])
def test_evaluate_writes_targets_and_predictions_hdf5(tmp_path, channel_list) -> None:
    class ChannelEncoder(torch.nn.Module):
        def forward(self, inputs):
            assert not self.training
            assert not torch.is_grad_enabled()
            assert isinstance(inputs, list)
            assert [value.dtype for value in inputs] == [torch.float32, torch.complex64, torch.float32]
            assert all(value.device.type == "cpu" for value in inputs)
            # Like correlation preprocessing, replace entries in the input list.
            inputs[0] = inputs[0] + 1
            return torch.stack((inputs[0].mean(dim=1), inputs[1].imag.mean(dim=1)), dim=-1)

    model = _identity_model(ChannelEncoder() if channel_list else None)
    batches = []
    for values in (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0]])):
        if channel_list:
            inputs = [
                (values[:, :1] - 1).expand(-1, 3).clone(),
                torch.complex(torch.ones(len(values), 5), values[:, 1:].expand(-1, 5)),
                torch.zeros(len(values), 7),
            ]
        else:
            inputs = values
        batches.append((inputs, values + 0.5, torch.arange(len(values))))

    def loader():
        yield from batches
        pytest.fail("Evaluation read beyond the requested number of examples")

    output_path = tmp_path / "eval" / "evaluation-step-0001.h5"
    metrics = evaluate(model, loader(), "cpu", num_examples=3, predictions_path=output_path)

    assert metrics == {"loss": pytest.approx(0.25), "prediction_loss": pytest.approx(0.25)}
    with h5py.File(output_path, "r") as handle:
        assert set(handle.keys()) == {"targets", "predictions"}
        torch.testing.assert_close(torch.from_numpy(handle["targets"][:]), torch.tensor([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]]))
        torch.testing.assert_close(torch.from_numpy(handle["predictions"][:]), torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    if channel_list:
        # Both passes must leave the loader's original channel list untouched.
        torch.testing.assert_close(batches[0][0][0], torch.tensor([[0.0] * 3, [2.0] * 3]))


def test_evaluate_rejects_mismatched_target_prediction_shapes(tmp_path) -> None:
    dataloader = [(torch.tensor([[1.0, 2.0]]), torch.tensor([[1.0]]), torch.tensor([0]))]
    with pytest.warns(UserWarning, match="target size"), pytest.raises(ValueError, match="same shape"):
        evaluate(_identity_model(), dataloader, "cpu", predictions_path=tmp_path / "bad.h5")


def test_evaluate_returns_standard_mse_alongside_configured_loss() -> None:
    class AbsoluteErrorModel(torch.nn.Module):
        def forward(self, inputs, targets):
            return torch.nn.functional.l1_loss(self.predict(inputs), targets)

        def predict(self, inputs):
            return inputs

    dataloader = [(torch.tensor([[1.0], [3.0]]), torch.tensor([[2.0], [5.0]]), torch.tensor([0, 1]))]
    metrics = evaluate(AbsoluteErrorModel(), dataloader, "cpu")
    assert metrics == {"loss": pytest.approx(1.5), "prediction_loss": pytest.approx(2.5)}


def test_evaluate_empty_loader_does_not_write_predictions(tmp_path) -> None:
    path = tmp_path / "empty.h5"
    assert evaluate(_identity_model(), [], "cpu", predictions_path=path) is None
    assert not path.exists()


def test_evaluate_accepts_positional_predictions_path(tmp_path) -> None:
    values = torch.ones(1, 2)
    path = tmp_path / "predictions.h5"
    metrics = evaluate(_identity_model(), [(values, values, torch.tensor([0]))], "cpu", path)
    assert metrics == {"loss": 0.0, "prediction_loss": 0.0}
    assert path.exists()
