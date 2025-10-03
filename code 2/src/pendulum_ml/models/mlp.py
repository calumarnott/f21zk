import torch.nn as nn
class MLP(nn.Module):
    """ Simple Multi-Layer Perceptron (MLP) model."""
    
    def __init__(self, in_dim=2, hidden=(128,128), out_dim=1, dropout=0.0):
        """
        Args:
            in_dim (int, optional): dimensionality of input. Defaults to 2.
            hidden (tuple, optional): tuple specifying hidden layer sizes. Defaults to (128, 128).
            out_dim (int, optional): dimensionality of output. Defaults to 1.
            dropout (float, optional): dropout rate (0.0 means no dropout). Defaults to 0.0.
        """
        super().__init__()
        
        layers=[]
        d=in_dim
        
        for h in hidden:
            
            layers += [nn.Linear(d,h), nn.ReLU()] # linear layer + ReLU activation
            
            if dropout>0: 
                layers += [nn.Dropout(dropout)]
                
            d=h
            
        layers += [nn.Linear(d,out_dim)] # final linear layer
        
        self.net = nn.Sequential(*layers) # create the sequential model
        
    def forward(self, x): 
        return self.net(x)
