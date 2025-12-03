import torch.nn as nn

from configs.base import Config


class SimpleNN(nn.Module):
    def __init__(self, cfg: Config):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(cfg.input_size, cfg.output_size, bias=False)

    def forward(self, x):
        out = self.fc1(x)
        return {"logits": out}
