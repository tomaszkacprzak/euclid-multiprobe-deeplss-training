import torch
import torch.nn as nn

class MSEHead(nn.Module):
    
    def __init__(self, embed_dim: int, num_targets: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_targets)
        self.mse_loss = nn.MSELoss()

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred = self.linear(z)
        return self.mse_loss(y_pred, y)

    @torch.no_grad()
    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)

class MSEModel(nn.Module):

    def __init__(self, encoder: nn.Module, loss: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.loss = loss

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = self.encoder(inputs)
        return self.loss(predictions, targets)

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)
