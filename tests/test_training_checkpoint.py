from __future__ import annotations

import pytest


def test_save_and_load_checkpoint_round_trip(tmp_path) -> None:
    torch = pytest.importorskip("torch")

    from euclid_multiprobe_deeplss_training.training import TrainingConfig, load_checkpoint, save_checkpoint

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    inputs = torch.tensor([[1.0, 2.0]])
    targets = torch.tensor([[3.0]])
    loss = torch.nn.MSELoss()(model(inputs), targets)
    loss.backward()
    optimizer.step()

    train_losses = [1.25, 0.75]
    val_losses = [0.95]
    config = TrainingConfig(records_pattern="records/*.tar", max_steps=2, learning_rate=0.01)
    checkpoint_path = tmp_path / "checkpoint.pt"

    saved_weight = model.weight.detach().clone()
    saved_bias = model.bias.detach().clone()
    save_checkpoint(checkpoint_path, model, optimizer, 2, config, train_losses, val_losses)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert set(checkpoint) == {
        "model_state_dict",
        "optimizer_state_dict",
        "step",
        "config",
        "train_losses",
        "val_losses",
    }
    assert checkpoint["step"] == 2
    assert checkpoint["train_losses"] == train_losses
    assert checkpoint["val_losses"] == val_losses
    assert checkpoint["config"]["records_pattern"] == "records/*.tar"

    with torch.no_grad():
        model.weight.fill_(0.0)
        model.bias.fill_(0.0)

    restored_step, restored_train_losses, restored_val_losses = load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        torch.device("cpu"),
    )

    assert restored_step == 2
    assert restored_train_losses == train_losses
    assert restored_val_losses == val_losses
    assert torch.equal(model.weight, saved_weight)
    assert torch.equal(model.bias, saved_bias)
