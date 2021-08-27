import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.net.basenet import BaseNetwork
from model.ops.norm import get_nonspade_norm_layer


class ConvEncoder(BaseNetwork):
    def __init__(self, opt):
        super(ConvEncoder, self).__init__()

        kw = 3
        pw = int(np.ceil((kw-1.0) / 2))
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        if opt.nef <= 0:
            nef = opt.ngf
        else:
            nef = opt.nef
        print('nef: {}'.format(nef))
        self.layer1 = norm_layer(nn.Conv2d(3, nef, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(nef, nef*2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(nef*2, nef*4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(nef*4, nef*8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(nef*8, nef*8, kw, stride=2, padding=pw))
        if opt.crop_size >= 256:
            self.layer6 = norm_layer(nn.Conv2d(nef*8, nef*8, kw, stride=2, padding=pw))

        self.s0 = s0 = 4
        self.fc_mu = nn.Linear(nef*8*s0*s0, opt.z_dim)
        self.fc_var = nn.Linear(nef*8*s0*s0, opt.z_dim)

        self.actvn = nn.LeakyReLU(2e-1, False)
        self.opt = opt

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256: # necessary? todo
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if self.opt.crop_size >= 256:
            x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar
