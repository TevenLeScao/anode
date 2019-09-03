from torch import nn
from anode.anode import odesolver_adjoint as odesolver


def make_odeblock(nt, method):

    class ODEBlock(nn.Module):

        def __init__(self, odefunc):
            super(ODEBlock, self).__init__()
            self.odefunc = odefunc
            self.options = {}
            self.options.update({'Nt': int(nt)})
            self.options.update({'method': method})

        def forward(self, x):
            out = odesolver(self.odefunc, x, self.options)
            return out

        @property
        def nfe(self):
            return self.odefunc.nfe

        @nfe.setter
        def nfe(self, value):
            self.odefunc.nfe = value

    return ODEBlock
