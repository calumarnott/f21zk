import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self,
                 in_dim=1,
                 channels=(64, 128, 128),
                 kernel_sizes=(5, 5, 3),
                 strides=(1, 1, 1),
                 pool_every=0,
                 batch_norm=True,
                 dropout=0.0,
                 head_hidden=(128,),
                 out_dim=1,
                 activation="relu"):
        """
        1D Convolutional Neural Network for time series data.
        """
        super().__init__()
        
        assert len(channels) == len(kernel_sizes) == len(strides), "channels, kernel_sizes, and strides must have the same length"
        
        act_fn = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()
        }[activation]
        
        blocks = []
        c_in = in_dim
        
        for i, (c_out, k, s) in enumerate(zip(channels, kernel_sizes, strides)):
            layers = [nn.Conv1d(c_in, c_out, kernel_size=k, stride=s, padding=k//2, bias=not batch_norm)]
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(c_out))
                
            layers.append(act_fn)
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
                
            if pool_every and pool_every > 0:
                layers.append(nn.MaxPool1d(kernel_size=pool_every))
                
            blocks += layers
            c_in = c_out
            
        self.feature_extractor = nn.Sequential(*blocks)
        
        self.head = []
        
        self.head.append(nn.AdaptiveAvgPool1d(1))  # Global average pooling
        self.head.append(nn.Flatten())
        last = channels[-1]
        for h in head_hidden:
            self.head.append(nn.Linear(last, h), act_fn)
            if dropout and dropout > 0.0:
                self.head.append(nn.Dropout(dropout))
            last = h
            
        self.head.append(nn.Linear(last, out_dim))
        self.head = nn.Sequential(*self.head)
        
    def forward(self, x):
        z = self.feature_extractor(x)
        y = self.head(z)
        return y
