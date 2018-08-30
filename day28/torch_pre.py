import torch.nn as nn
import torch

##############################################
# Example - 1
# Linear Layer
linear = nn.Linear(in_features = 2, out_features = 5, bias = True)
in_feat = torch.FloatTensor([1, 2])
print linear(in_feat)

##############################################
# Example - 2
# Sequential Stacking
s = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.ReLU(),
    nn.Linear(in_features=5, out_features=20),
    nn.ReLU(),
    nn.Linear(in_features=20, out_features=10),
    nn.Dropout(p = 0.3),
    nn.Softmax(dim = 1)   # across data dimension dim = 0; batch sample
)
print s

print s(torch.FloatTensor([[1,2]]))

##############################################
# Example - 3
class OurModule(nn.Module):

    def __init__(self, inp, classes, dropout):
        super(OurModule, self).__init__()
        self.pipeline = nn.Sequential(
            nn.Linear(in_features=inp, out_features=5),
            nn.ReLU(),
            nn.Linear(in_features=5, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=classes),
            nn.Dropout(p=dropout),
            nn.Softmax()
        )
    
    def forward(self, x):
        return self.pipeline(x)

if __name__ == '__main__':
    net = OurModule(inp=2, classes=3, dropout=0.3)
    data = torch.FloatTensor([[2, 3]])
    output = net(data)
    print net
    print output
