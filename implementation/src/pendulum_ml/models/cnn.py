import torch
import torch.nn as nn

def _make_act(name: str) -> nn.Module:
    table = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    return table.get(name, nn.ReLU)()

class CNN1D(nn.Module):
    """
    Simple 1D CNN for time-series control policies.

    Expects input shaped [B, C, W] (batch, channels/features, window).
    Uses small kernels, no pooling or dropout by default (verification-friendly).
    """
    def __init__(self,
                 in_dim=1,                         # input channels (C)
                 channels=(64, 64, 64),            # conv feature maps per block
                 kernel_sizes=(2, 3, 3),           # small kernels; RF covers W=8 well
                 strides=(1, 1, 1),
                 pool_every=0,                     # keep 0 for W small; kept for API parity
                 batch_norm=False,                 # keep False for verification
                 dropout=0.0,                      # keep 0.0 for verification
                 head_hidden=(64,),
                 out_dim=1,                        # number of control outputs
                 activation="relu"):
        super().__init__()
        assert len(channels) == len(kernel_sizes) == len(strides), \
            "channels, kernel_sizes, and strides must have the same length"

        # ---- Feature extractor (Conv1d stack) ----
        blocks = []
        c_in = in_dim
        for c_out, k, s in zip(channels, kernel_sizes, strides):
            # "same"-length padding for stride=1
            pad = (k // 2) if s == 1 else 0
            layers = [nn.Conv1d(c_in, c_out, kernel_size=k, stride=s, padding=pad, bias=not batch_norm)]
            if batch_norm:
                layers.append(nn.BatchNorm1d(c_out))
            layers.append(_make_act(activation))
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            if pool_every and pool_every > 0:
                layers.append(nn.MaxPool1d(kernel_size=pool_every))
            blocks.extend(layers)
            c_in = c_out
        self.feature_extractor = nn.Sequential(*blocks)

        # ---- Head (global squeeze + MLP) ----
        head = []
        last = channels[-1]
        for h in head_hidden:
            head += [nn.Linear(last, h), _make_act(activation)]
            if dropout and dropout > 0.0:
                head.append(nn.Dropout(dropout))
            last = h
        head.append(nn.Linear(last, out_dim))
        self.head = nn.Sequential(*head)

    def forward(self, x):
        z = self.feature_extractor(x)   # [B, C_last, W]
        z_last = z[:, :, -1]            # take most recent timestep -> [B, C_last]
        y = self.head(z_last)           # [B, out_dim]
        return y