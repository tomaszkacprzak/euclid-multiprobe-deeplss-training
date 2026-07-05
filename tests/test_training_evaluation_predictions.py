from __future__ import annotations

import pytest

h5py = pytest.importorskip("h5py")


def test_evaluate_writes_targets_and_predictions_hdf5(tmp_path) -> None:
    torch = pytest.importorskip("torch")

    from euclid_multiprobe_deeplss_training.training import evaluate

    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
        model.bias.zero_()

    dataloader = [
        (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[1.5, 2.5], [3.5, 4.5]])),
        (torch.tensor([[5.0, 6.0]]), torch.tensor([[5.5, 6.5]])),
    ]
    output_path = tmp_path / "eval" / "evaluation-epoch-0001.h5"

    loss = evaluate(model, dataloader, torch.nn.MSELoss(), torch.device("cpu"), output_path)

    assert loss == pytest.approx(0.25)
    assert output_path.exists()
    with h5py.File(output_path, "r") as handle:
        assert set(handle.keys()) == {"targets", "predictions"}
        assert handle["targets"].shape == (3, 2)
        assert handle["predictions"].shape == (3, 2)
        assert handle["targets"].shape == handle["predictions"].shape
        torch.testing.assert_close(torch.from_numpy(handle["targets"][:]), torch.tensor([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]]))
        torch.testing.assert_close(torch.from_numpy(handle["predictions"][:]), torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))


def test_evaluate_rejects_mismatched_target_prediction_shapes(tmp_path) -> None:
    torch = pytest.importorskip("torch")

    from euclid_multiprobe_deeplss_training.training import evaluate

    dataloader = [(torch.tensor([[1.0, 2.0]]), torch.tensor([[1.0]]))]

    with pytest.raises(ValueError, match="same shape"):
        evaluate(torch.nn.Linear(2, 2), dataloader, torch.nn.MSELoss(), torch.device("cpu"), tmp_path / "bad.h5")


def test_evaluate_returns_standard_mse_alongside_configured_loss() -> None:
    torch = pytest.importorskip("torch")

    from euclid_multiprobe_deeplss_training.training import evaluate

    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.zero_()

    dataloader = [(torch.tensor([[1.0], [3.0]]), torch.tensor([[2.0], [5.0]]))]

    metrics = evaluate(
        model,
        dataloader,
        torch.nn.L1Loss(),
        torch.device("cpu"),
        return_metrics=True,
    )

    assert metrics == {"loss": pytest.approx(1.5), "mse_loss": pytest.approx(2.5)}
