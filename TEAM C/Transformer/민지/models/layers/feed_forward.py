import torch
import torch.nn as nn
'''
This consists of two linear transformations with a ReLU activation in between.
Another way of describing this is as two convolutions with kernel size 1.
The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality
d_ff=2048.

We apply dropout to the output of each sub-layer, before it is added to the
sub-layer input and normalized. we use a rate of P_drop = 0.1
'''
class FeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(d_model, hidden) # 512 2048
        self.l2 = nn.Linear(hidden, d_model) # 2048 512
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.dropout(x)
        return x


