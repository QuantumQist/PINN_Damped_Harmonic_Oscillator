from torch import nn

class PINN(nn.Module):
    def __init__(self, device):
        super(PINN, self).__init__()

        self.best_loss = float("inf")

        self.mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        ).to(device)

        self.device = device

    def forward(self, t):
        return self.mlp(t)