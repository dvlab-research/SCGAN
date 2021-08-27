import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import util.func as util
from model.net.basenet import BaseNetwork
from model.ops.layer import Blur


class MultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
                                            'model.net.discriminator')
        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


from model.ops.norm import get_nonspade_norm_layer


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc(opt)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, opt):
        input_nc = opt.label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]

# Feature-Pyramid Semantics Embedding Discriminator
class FPSEDiscriminator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ndf
        input_nc = 3
        label_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        self.enc1 = nn.Sequential(norm_layer(nn.Conv2d(input_nc, nf, kernel_size=3, stride=2, padding=1)),
                                  nn.LeakyReLU(2e-1, True))
        self.enc2 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf*2, kernel_size=3, stride=2, padding=1)),
                                  nn.LeakyReLU(2e-1, True))
        self.enc3 = nn.Sequential(norm_layer(nn.Conv2d(nf*2, nf*4, kernel_size=3, stride=2, padding=1)),
                                  nn.LeakyReLU(2e-1, True))
        self.enc4 = nn.Sequential(norm_layer(nn.Conv2d(nf*4, nf*8, kernel_size=3, stride=2, padding=1)),
                                  nn.LeakyReLU(2e-1, True))
        self.enc5 = nn.Sequential(norm_layer(nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=2, padding=1)),
                                  nn.LeakyReLU(2e-1, True))

        # top-down pathway
        self.lat2 = nn.Sequential(norm_layer(nn.Conv2d(nf*2, nf*4, kernel_size=1)),
                                  nn.LeakyReLU(2e-1, True))
        self.lat3 = nn.Sequential(norm_layer(nn.Conv2d(nf*4, nf*4, kernel_size=1)),
                                  nn.LeakyReLU(2e-1, True))
        self.lat4 = nn.Sequential(norm_layer(nn.Conv2d(nf*8, nf*4, kernel_size=1)),
                                  nn.LeakyReLU(2e-1, True))
        self.lat5 = nn.Sequential(norm_layer(nn.Conv2d(nf*8, nf*4, kernel_size=1)),
                                  nn.LeakyReLU(2e-1, True))

        # upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        # final layers
        self.final2 = nn.Sequential(norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1)),
                                    nn.LeakyReLU(2e-1, True))
        self.final3 = nn.Sequential(norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1)),
                                    nn.LeakyReLU(2e-1, True))
        self.final4 = nn.Sequential(norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1)),
                                    nn.LeakyReLU(2e-1, True))

        # true / false prediction and semantic alignment prediction
        self.tf = nn.Conv2d(nf*2, 1, kernel_size=1)
        self.seg = nn.Conv2d(nf*2, nf*2, kernel_size=1)
        self.embedding = nn.Conv2d(label_nc, nf*2, kernel_size=1)
        self.blur = Blur() if self.opt.smooth else None

    def forward(self, fake_and_real_img, segmap):
        # bottom-up pathway
        feat11 = self.enc1(fake_and_real_img)
        feat12 = self.enc2(feat11)
        feat13 = self.enc3(feat12)
        feat14 = self.enc4(feat13)
        feat15 = self.enc5(feat14)
        feat15 = self.blur(feat15) if self.opt.smooth else feat15

        # top-down pathway and lateral connections
        feat25 = self.lat5(feat15)
        feat24 = self.up(feat25) + self.lat4(feat14)
        feat23 = self.up(feat24) + self.lat3(feat13)
        feat22 = self.up(feat23) + self.lat2(feat12)

        # final prediction layers
        feat32 = self.final2(feat22)
        feat33 = self.final3(feat23)
        feat34 = self.final4(feat24)

        # patch-based true / false prediction
        pred2 = self.tf(feat32)
        pred3 = self.tf(feat33)
        pred4 = self.tf(feat34)
        seg2 = self.seg(feat32)
        seg3 = self.seg(feat33)
        seg4 = self.seg(feat34)

        # intermediate features for discriminator feature matching losss
        feats = [feat12, feat13, feat14, feat15]

        # segmentation map embedding
        segembed = self.embedding(segmap)
        segembed = F.avg_pool2d(segembed, kernel_size=2, stride=2)
        segembed2 = F.avg_pool2d(segembed, kernel_size=2, stride=2)
        segembed3 = F.avg_pool2d(segembed2, kernel_size=2, stride=2)
        segembed4 = F.avg_pool2d(segembed3, kernel_size=2, stride=2)

        # semantics embedding discriminator score
        pred2 += torch.mul(segembed2, seg2).sum(dim=1, keepdim=True)
        pred3 += torch.mul(segembed3, seg3).sum(dim=1, keepdim=True)
        pred4 += torch.mul(segembed4, seg4).sum(dim=1, keepdim=True)

        # concat results from multiple resolutions
        results = [pred2, pred3, pred4]
        return [feats, results]


from torch.nn.utils import spectral_norm


# Feature-Pyramid Semantics Embedding Discriminator Modified
class FPSEMDiscriminator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ndf
        input_nc = 3
        label_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)

        self.enc1 = nn.Sequential(norm_layer(nn.Conv2d(input_nc, nf, kernel_size=3, stride=2, padding=1)),
                                  nn.LeakyReLU(2e-1, True))
        self.enc2 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf*2, kernel_size=3, stride=2, padding=1)),
                                  nn.LeakyReLU(2e-1, True))
        self.enc3 = nn.Sequential(norm_layer(nn.Conv2d(nf*2, nf*4, kernel_size=3, stride=2, padding=1)),
                                  nn.LeakyReLU(2e-1, True))
        self.enc4 = nn.Sequential(norm_layer(nn.Conv2d(nf*4, nf*8, kernel_size=3, stride=2, padding=1)),
                                  nn.LeakyReLU(2e-1, True))
        self.enc5 = nn.Sequential(norm_layer(nn.Conv2d(nf*8, nf*16, kernel_size=3, stride=2, padding=1)),
                                  nn.LeakyReLU(2e-1, True))
        self.enc6 = nn.Sequential(norm_layer(nn.Conv2d(nf * 16, nf * 16, kernel_size=3, stride=2, padding=1)),
                                  nn.LeakyReLU(2e-1, True))

        # top-down pathway
        # self.lat2 = nn.Sequential(norm_layer(nn.Conv2d(nf*2, nf*4, kernel_size=1)),
        #                           nn.LeakyReLU(2e-1, True))
        self.lat3 = nn.Sequential(norm_layer(nn.Conv2d(nf*2, nf*4, kernel_size=1)),
                                  nn.LeakyReLU(2e-1, True))
        self.lat4 = nn.Sequential(norm_layer(nn.Conv2d(nf*8, nf*4, kernel_size=1)),
                                  nn.LeakyReLU(2e-1, True))
        self.lat5 = nn.Sequential(norm_layer(nn.Conv2d(nf*16, nf*4, kernel_size=1)),
                                  nn.LeakyReLU(2e-1, True))
        self.lat6 = nn.Sequential(norm_layer(nn.Conv2d(nf * 16, nf * 4, kernel_size=1)),
                                  nn.LeakyReLU(2e-1, True))

        # upsampling
        self.up = nn.Upsample(scale_factor=4, mode='bilinear')

        # final layers
        self.final2 = nn.Sequential(norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1)),
                                    nn.LeakyReLU(2e-1, True))
        self.final3 = nn.Sequential(norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1)),
                                    nn.LeakyReLU(2e-1, True))
        self.final4 = nn.Sequential(norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1)),
                                    nn.LeakyReLU(2e-1, True))
        self.final5 = nn.Sequential(norm_layer(nn.Conv2d(nf * 4, nf * 2, kernel_size=3, padding=1)),
                                    nn.LeakyReLU(2e-1, True))

        # true / false prediction and semantic alignment prediction
        self.tf1 = spectral_norm(nn.Conv2d(nf*2, 1, kernel_size=1))
        self.seg = None
        self.embedding = spectral_norm(nn.Conv2d(label_nc, nf*2, kernel_size=1))
        self.blur = Blur() if self.opt.smooth else None

    def forward(self, fake_and_real_img, segmap):
        # bottom-up pathway
        feat11 = self.enc1(fake_and_real_img) # 128
        feat12 = self.enc2(feat11) # 64
        feat13 = self.enc3(feat12) # 32
        feat14 = self.enc4(feat13) # 16
        feat15 = self.enc5(feat14) # 8
        feat15 = self.blur(feat15) if self.opt.smooth else feat15
        feat16 = self.enc6(feat15) # 4
        feat17 = torch.mean(feat16, (2, 3), keepdim=True) # global pooling
        # pred1 = self.l6(feat17)
        # pred1 = pred1.view(pred1.size(0), 1, 1, 1)

        # top-down pathway and lateral connections
        feat26 = self.lat6(feat17)
        # print('26,', feat26.shape, feat16.shape)
        feat25 = self.lat5(feat16) + feat26 # 4
        feat24 = self.up(feat25) + self.lat4(feat14) # 16
        # feat23 = self.up(feat24) + self.lat4(feat13) # 64
        feat23 = self.up(feat24) + self.lat3(feat12) # 64

        # final prediction layers
        feat32 = self.final2(feat23)
        feat33 = self.final3(feat24)
        feat34 = self.final4(feat25)
        feat35 = self.final5(feat26)

        # patch-based true / false prediction
        pred2 = self.tf1(feat32)
        pred3 = self.tf1(feat33)
        pred4 = self.tf1(feat34)
        pred1 = self.tf1(feat35)
        seg2 = self.seg(feat32) if self.seg else feat32
        seg3 = self.seg(feat33) if self.seg else feat33
        seg4 = self.seg(feat34) if self.seg else feat34
        seg5 = self.seg(feat35) if self.seg else feat35

        # intermediate features for discriminator feature matching losses
        feats = [feat12, feat13, feat14, feat15]

        # segmentation map embedding
        segembed = self.embedding(segmap) # 256
        segembed2 = F.avg_pool2d(segembed, kernel_size=4, stride=4) # 64
        segembed3 = F.avg_pool2d(segembed2, kernel_size=4, stride=4) # 16
        segembed4 = F.avg_pool2d(segembed3, kernel_size=4, stride=4) # 4
        segembed5 = F.adaptive_avg_pool2d(segembed4, (1, 1))
        # segembed5 = F.avg_pool2d(segembed4, kernel_size=4, stride=4)  # 1

        # semantics embedding discriminator score
        pred2 += torch.mul(segembed2, seg2).sum(dim=1, keepdim=True)
        pred3 += torch.mul(segembed3, seg3).sum(dim=1, keepdim=True)
        pred4 += torch.mul(segembed4, seg4).sum(dim=1, keepdim=True)
        pred1 += torch.mul(segembed5, seg5).sum(dim=1, keepdim=True)

        # concat results from multiple resolutions
        results = [pred2, pred3, pred4, pred1]
        return [feats, results]


from model.ops.local import SpatialCondConv2dFast


class RoutingDiscriminator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ndf
        input_nc = 3
        label_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

        num_experts_conv = opt.num_experts_conv
        spectral = True if 'spectral' in opt.norm_D else False
        norm_D = opt.norm_D if spectral is False else opt.norm_D[8:]
        norm_layer = get_nonspade_norm_layer(opt, norm_D)

        self.enc1 = SpatialCondConv2dFast(in_channels=input_nc, out_channels=nf, stride=2,
                                                                   kernel_size=3, num_experts=num_experts_conv,
                                                                   spectral=spectral)
        self.enc11 = nn.Sequential(nn.InstanceNorm2d(nf, affine=False), nn.LeakyReLU(2e-1, True))

        self.enc2 = SpatialCondConv2dFast(in_channels=nf, out_channels=nf*2, stride=2,
                                          kernel_size=3, num_experts=num_experts_conv,
                                          spectral=spectral)
        self.enc21 = nn.Sequential(nn.InstanceNorm2d(nf*2, affine=False), nn.LeakyReLU(2e-1, True))

        self.enc3 = SpatialCondConv2dFast(in_channels=nf*2, out_channels=nf*4, stride=2,
                                          kernel_size=3, num_experts=num_experts_conv,
                                          spectral=spectral)
        self.enc31 = nn.Sequential(nn.InstanceNorm2d(nf*4, affine=False), nn.LeakyReLU(2e-1, True))

        self.enc4 = SpatialCondConv2dFast(in_channels=nf*4, out_channels=nf*8, stride=2,
                                          kernel_size=3, num_experts=num_experts_conv,
                                          spectral=spectral)
        self.enc41 = nn.Sequential(nn.InstanceNorm2d(nf*8, affine=False), nn.LeakyReLU(2e-1, True))

        self.enc5 = SpatialCondConv2dFast(in_channels=nf*8, out_channels=nf*8, stride=2,
                                          kernel_size=3, num_experts=num_experts_conv,
                                          spectral=spectral)
        self.enc51 = nn.Sequential(nn.InstanceNorm2d(nf*8, affine=False), nn.LeakyReLU(2e-1, True))


        # top-down pathway
        self.lat2 = nn.Sequential(norm_layer(nn.Conv2d(nf*2, nf*4, kernel_size=1)),
                                  nn.LeakyReLU(2e-1, True))
        self.lat3 = nn.Sequential(norm_layer(nn.Conv2d(nf*4, nf*4, kernel_size=1)),
                                  nn.LeakyReLU(2e-1, True))
        self.lat4 = nn.Sequential(norm_layer(nn.Conv2d(nf*8, nf*4, kernel_size=1)),
                                  nn.LeakyReLU(2e-1, True))
        self.lat5 = nn.Sequential(norm_layer(nn.Conv2d(nf*8, nf*4, kernel_size=1)),
                                  nn.LeakyReLU(2e-1, True))

        # upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        # final layers
        self.final2 = nn.Sequential(norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1)),
                                    nn.LeakyReLU(2e-1, True))
        self.final3 = nn.Sequential(norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1)),
                                    nn.LeakyReLU(2e-1, True))
        self.final4 = nn.Sequential(norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1)),
                                    nn.LeakyReLU(2e-1, True))

        # true / false prediction and semantic alignment prediction
        self.tf = nn.Conv2d(nf*2, 1, kernel_size=1)
        self.seg = nn.Conv2d(nf*2, nf*2, kernel_size=1)
        self.embedding = nn.Conv2d(label_nc, nf*2, kernel_size=1)

    def forward(self, fake_and_real_img, segmap, routing_weights):
        # bottom-up pathway
        feat11 = self.enc1(fake_and_real_img, routing_weights[4].repeat(2, 1, 1, 1))
        feat11 = self.enc11(feat11)
        feat12 = self.enc2(feat11, routing_weights[3].repeat(2, 1, 1, 1))
        feat12 = self.enc21(feat12)
        feat13 = self.enc3(feat12, routing_weights[2].repeat(2, 1, 1, 1))
        feat13 = self.enc31(feat13)
        feat14 = self.enc4(feat13, routing_weights[1].repeat(2, 1, 1, 1))
        feat14 = self.enc41(feat14)
        feat15 = self.enc5(feat14, routing_weights[0].repeat(2, 1, 1, 1))
        feat15 = self.enc51(feat15)

        # top-down pathway and lateral connections
        feat25 = self.lat5(feat15)
        feat24 = self.up(feat25) + self.lat4(feat14)
        feat23 = self.up(feat24) + self.lat3(feat13)
        feat22 = self.up(feat23) + self.lat2(feat12)

        # final prediction layers
        feat32 = self.final2(feat22)
        feat33 = self.final3(feat23)
        feat34 = self.final4(feat24)

        # patch-based true / false prediction
        pred2 = self.tf(feat32)
        pred3 = self.tf(feat33)
        pred4 = self.tf(feat34)
        seg2 = self.seg(feat32)
        seg3 = self.seg(feat33)
        seg4 = self.seg(feat34)

        # intermediate features for discriminator feature matching losss
        feats = [feat12, feat13, feat14, feat15]

        # segmentation map embedding
        segembed = self.embedding(segmap)
        segembed = F.avg_pool2d(segembed, kernel_size=2, stride=2)
        segembed2 = F.avg_pool2d(segembed, kernel_size=2, stride=2)
        segembed3 = F.avg_pool2d(segembed2, kernel_size=2, stride=2)
        segembed4 = F.avg_pool2d(segembed3, kernel_size=2, stride=2)

        # semantics embedding discriminator score
        pred2 += torch.mul(segembed2, seg2).sum(dim=1, keepdim=True)
        pred3 += torch.mul(segembed3, seg3).sum(dim=1, keepdim=True)
        pred4 += torch.mul(segembed4, seg4).sum(dim=1, keepdim=True)

        # concat results from multiple resolutions
        results = [pred2, pred3, pred4]
        return [feats, results]


class SRoutingDiscriminator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ndf
        input_nc = 3
        label_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

        num_experts_conv = opt.num_experts_conv
        spectral = True if 'spectral' in opt.norm_D else False
        norm_D = opt.norm_D if spectral is False else opt.norm_D[8:]
        norm_layer = get_nonspade_norm_layer(opt, norm_D)

        self.enc1 = SpatialCondConv2dFast(in_channels=input_nc, out_channels=nf, stride=2,
                                                                   kernel_size=3, num_experts=num_experts_conv,
                                                                   spectral=spectral)
        self.enc11 = nn.Sequential(nn.InstanceNorm2d(nf, affine=False), nn.LeakyReLU(2e-1, True))

        self.enc2 = SpatialCondConv2dFast(in_channels=nf, out_channels=nf*2, stride=2,
                                          kernel_size=3, num_experts=num_experts_conv,
                                          spectral=spectral)
        self.enc21 = nn.Sequential(nn.InstanceNorm2d(nf*2, affine=False), nn.LeakyReLU(2e-1, True))

        self.enc3 = SpatialCondConv2dFast(in_channels=nf*2, out_channels=nf*4, stride=2,
                                          kernel_size=3, num_experts=num_experts_conv,
                                          spectral=spectral)
        self.enc31 = nn.Sequential(nn.InstanceNorm2d(nf*4, affine=False), nn.LeakyReLU(2e-1, True))

        self.enc4 = SpatialCondConv2dFast(in_channels=nf*4, out_channels=nf*8, stride=2,
                                          kernel_size=3, num_experts=num_experts_conv,
                                          spectral=spectral)
        self.enc41 = nn.Sequential(nn.InstanceNorm2d(nf*8, affine=False), nn.LeakyReLU(2e-1, True))

        self.enc5 = SpatialCondConv2dFast(in_channels=nf*8, out_channels=nf*8, stride=2,
                                          kernel_size=3, num_experts=num_experts_conv,
                                          spectral=spectral)
        self.enc51 = nn.Sequential(nn.InstanceNorm2d(nf*8, affine=False), nn.LeakyReLU(2e-1, True))


        # top-down pathway
        self.lat2 = nn.Sequential(norm_layer(nn.Conv2d(nf*2, nf*4, kernel_size=1)),
                                  nn.LeakyReLU(2e-1, True))
        self.lat3 = nn.Sequential(norm_layer(nn.Conv2d(nf*4, nf*4, kernel_size=1)),
                                  nn.LeakyReLU(2e-1, True))
        self.lat4 = nn.Sequential(norm_layer(nn.Conv2d(nf*8, nf*4, kernel_size=1)),
                                  nn.LeakyReLU(2e-1, True))
        self.lat5 = nn.Sequential(norm_layer(nn.Conv2d(nf*8, nf*4, kernel_size=1)),
                                  nn.LeakyReLU(2e-1, True))

        # upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        # final layers
        self.final2 = nn.Sequential(norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1)),
                                    nn.LeakyReLU(2e-1, True))
        self.final3 = nn.Sequential(norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1)),
                                    nn.LeakyReLU(2e-1, True))
        self.final4 = nn.Sequential(norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1)),
                                    nn.LeakyReLU(2e-1, True))

        # true / false prediction and semantic alignment prediction
        self.tf = nn.Conv2d(nf*2, 1, kernel_size=1)
        self.seg = nn.Conv2d(nf*2, nf*2, kernel_size=1)
        self.embedding = nn.Conv2d(label_nc, nf*2, kernel_size=1)
        if opt.weight_type == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        elif opt.weight_type == 'softmax':
            self.act_fn = nn.Softmax2d()
        else:
            self.act_fn = lambda x: x

        if self.opt.srouting_detach:
            print('detach the gradients of the vae from the discriminator')

        self.blur = Blur() if self.opt.smooth else None

    def forward(self, fake_and_real_img, segmap, routing_weights):
        # bottom-up pathway
        if self.opt.srouting_detach:
            routing_weights = [self.act_fn(F.adaptive_avg_pool2d(x.detach(), (1, 1))*self.opt.temperature) for x in
                               routing_weights[:5]]
        else:
            routing_weights = [self.act_fn(F.adaptive_avg_pool2d(x, (1, 1)) * self.opt.temperature) for x in
                               routing_weights[:5]]

        feat11 = self.enc1(fake_and_real_img, routing_weights[4].repeat(2, 1, 1, 1))

        feat11 = self.blur(feat11) if self.opt.smooth else feat11

        feat11 = self.enc11(feat11)
        feat12 = self.enc2(feat11, routing_weights[3].repeat(2, 1, 1, 1))

        feat12 = self.blur(feat12) if self.opt.smooth else feat12

        feat12 = self.enc21(feat12)
        feat13 = self.enc3(feat12, routing_weights[2].repeat(2, 1, 1, 1))

        feat13 = self.blur(feat13) if self.opt.smooth else feat13

        feat13 = self.enc31(feat13)
        feat14 = self.enc4(feat13, routing_weights[1].repeat(2, 1, 1, 1))

        feat14 = self.blur(feat14) if self.opt.smooth else feat14

        feat14 = self.enc41(feat14)
        feat15 = self.enc5(feat14, routing_weights[0].repeat(2, 1, 1, 1))

        feat15 = self.blur(feat15) if self.opt.smooth else feat15

        feat15 = self.enc51(feat15)

        # top-down pathway and lateral connections
        feat25 = self.lat5(feat15)
        feat24 = self.up(feat25) + self.lat4(feat14)
        feat23 = self.up(feat24) + self.lat3(feat13)
        feat22 = self.up(feat23) + self.lat2(feat12)

        # final prediction layers
        feat32 = self.final2(feat22)
        feat33 = self.final3(feat23)
        feat34 = self.final4(feat24)

        # patch-based true / false prediction
        pred2 = self.tf(feat32)
        pred3 = self.tf(feat33)
        pred4 = self.tf(feat34)
        seg2 = self.seg(feat32)
        seg3 = self.seg(feat33)
        seg4 = self.seg(feat34)

        # intermediate features for discriminator feature matching losss
        feats = [feat12, feat13, feat14, feat15]

        # segmentation map embedding
        segembed = self.embedding(segmap)
        segembed = F.avg_pool2d(segembed, kernel_size=2, stride=2)
        segembed2 = F.avg_pool2d(segembed, kernel_size=2, stride=2)
        segembed3 = F.avg_pool2d(segembed2, kernel_size=2, stride=2)
        segembed4 = F.avg_pool2d(segembed3, kernel_size=2, stride=2)

        # semantics embedding discriminator score
        pred2 += torch.mul(segembed2, seg2).sum(dim=1, keepdim=True)
        pred3 += torch.mul(segembed3, seg3).sum(dim=1, keepdim=True)
        pred4 += torch.mul(segembed4, seg4).sum(dim=1, keepdim=True)

        # concat results from multiple resolutions
        results = [pred2, pred3, pred4]
        return [feats, results]

class SelfRoutingDiscriminator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ndf
        input_nc = 3
        label_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

        num_experts_conv = opt.num_experts_conv
        spectral = True if 'spectral' in opt.norm_D else False
        norm_D = opt.norm_D if spectral is False else opt.norm_D[8:]
        norm_layer = get_nonspade_norm_layer(opt, norm_D)

        self.enc1 = nn.Sequential(norm_layer(nn.Conv2d(input_nc, nf, kernel_size=3, stride=2, padding=1)),
                                  nn.LeakyReLU(2e-1, True))

        self.r_adjust1 = nn.Conv2d(nf, num_experts_conv, kernel_size=1, stride=1)

        self.enc2 = SpatialCondConv2dFast(in_channels=nf, out_channels=nf*2, stride=2,
                                          kernel_size=3, num_experts=num_experts_conv,
                                          spectral=spectral)
        self.enc21 = nn.Sequential(nn.InstanceNorm2d(nf*2, affine=False), nn.LeakyReLU(2e-1, True))

        self.r_adjust2 = nn.Conv2d(nf*2, num_experts_conv, kernel_size=1, stride=1)

        self.enc3 = SpatialCondConv2dFast(in_channels=nf*2, out_channels=nf*4, stride=2,
                                          kernel_size=3, num_experts=num_experts_conv,
                                          spectral=spectral)
        self.enc31 = nn.Sequential(nn.InstanceNorm2d(nf*4, affine=False), nn.LeakyReLU(2e-1, True))

        self.r_adjust3 = nn.Conv2d(nf*4, num_experts_conv, kernel_size=1, stride=1)

        self.enc4 = SpatialCondConv2dFast(in_channels=nf*4, out_channels=nf*8, stride=2,
                                          kernel_size=3, num_experts=num_experts_conv,
                                          spectral=spectral)
        self.enc41 = nn.Sequential(nn.InstanceNorm2d(nf*8, affine=False), nn.LeakyReLU(2e-1, True))

        self.r_adjust4 = nn.Conv2d(nf*8, num_experts_conv, kernel_size=1, stride=1)

        self.enc5 = SpatialCondConv2dFast(in_channels=nf*8, out_channels=nf*8, stride=2,
                                          kernel_size=3, num_experts=num_experts_conv,
                                          spectral=spectral)
        self.enc51 = nn.Sequential(nn.InstanceNorm2d(nf*8, affine=False), nn.LeakyReLU(2e-1, True))


        # top-down pathway
        self.lat2 = nn.Sequential(norm_layer(nn.Conv2d(nf*2, nf*4, kernel_size=1)),
                                  nn.LeakyReLU(2e-1, True))
        self.lat3 = nn.Sequential(norm_layer(nn.Conv2d(nf*4, nf*4, kernel_size=1)),
                                  nn.LeakyReLU(2e-1, True))
        self.lat4 = nn.Sequential(norm_layer(nn.Conv2d(nf*8, nf*4, kernel_size=1)),
                                  nn.LeakyReLU(2e-1, True))
        self.lat5 = nn.Sequential(norm_layer(nn.Conv2d(nf*8, nf*4, kernel_size=1)),
                                  nn.LeakyReLU(2e-1, True))

        # upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        # final layers
        self.final2 = nn.Sequential(norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1)),
                                    nn.LeakyReLU(2e-1, True))
        self.final3 = nn.Sequential(norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1)),
                                    nn.LeakyReLU(2e-1, True))
        self.final4 = nn.Sequential(norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1)),
                                    nn.LeakyReLU(2e-1, True))

        # true / false prediction and semantic alignment prediction
        self.tf = nn.Conv2d(nf*2, 1, kernel_size=1)
        self.seg = nn.Conv2d(nf*2, nf*2, kernel_size=1)
        self.embedding = nn.Conv2d(label_nc, nf*2, kernel_size=1)
        if opt.weight_type == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        elif opt.weight_type == 'softmax':
            self.act_fn = nn.Softmax2d()
        else:
            self.act_fn = lambda x: x

        self.blur = Blur() if self.opt.smooth else None

    def forward(self, fake_and_real_img, segmap, routing_weights):

        feat11 = self.enc1(fake_and_real_img)

        # feat11 = self.blur(feat11) if self.opt.smooth else feat11

        routing_weights1 = self.act_fn(F.adaptive_avg_pool2d(self.r_adjust1(feat11), (1, 1))*self.opt.temperature)

        feat12 = self.enc2(feat11, routing_weights1)

        # feat12 = self.blur(feat12) if self.opt.smooth else feat12

        feat12 = self.enc21(feat12)
        routing_weights2 = self.act_fn(F.adaptive_avg_pool2d(self.r_adjust2(feat12), (1, 1)) * self.opt.temperature)
        feat13 = self.enc3(feat12, routing_weights2)

        # feat13 = self.blur(feat13) if self.opt.smooth else feat13

        feat13 = self.enc31(feat13)
        routing_weights3 = self.act_fn(F.adaptive_avg_pool2d(self.r_adjust3(feat13), (1, 1)) * self.opt.temperature)
        feat14 = self.enc4(feat13, routing_weights3)

        # feat14 = self.blur(feat14) if self.opt.smooth else feat14

        feat14 = self.enc41(feat14)
        routing_weights4 = self.act_fn(F.adaptive_avg_pool2d(self.r_adjust4(feat14), (1, 1)) * self.opt.temperature)
        feat15 = self.enc5(feat14, routing_weights4)

        feat15 = self.blur(feat15) if self.opt.smooth else feat15

        feat15 = self.enc51(feat15)

        # top-down pathway and lateral connections
        feat25 = self.lat5(feat15)
        feat24 = self.up(feat25) + self.lat4(feat14)
        feat23 = self.up(feat24) + self.lat3(feat13)
        feat22 = self.up(feat23) + self.lat2(feat12)

        # final prediction layers
        feat32 = self.final2(feat22)
        feat33 = self.final3(feat23)
        feat34 = self.final4(feat24)

        # patch-based true / false prediction
        pred2 = self.tf(feat32)
        pred3 = self.tf(feat33)
        pred4 = self.tf(feat34)
        seg2 = self.seg(feat32)
        seg3 = self.seg(feat33)
        seg4 = self.seg(feat34)

        # intermediate features for discriminator feature matching losss
        feats = [feat12, feat13, feat14, feat15]

        # segmentation map embedding
        segembed = self.embedding(segmap)
        segembed = F.avg_pool2d(segembed, kernel_size=2, stride=2)
        segembed2 = F.avg_pool2d(segembed, kernel_size=2, stride=2)
        segembed3 = F.avg_pool2d(segembed2, kernel_size=2, stride=2)
        segembed4 = F.avg_pool2d(segembed3, kernel_size=2, stride=2)

        # semantics embedding discriminator score
        pred2 += torch.mul(segembed2, seg2).sum(dim=1, keepdim=True)
        pred3 += torch.mul(segembed3, seg3).sum(dim=1, keepdim=True)
        pred4 += torch.mul(segembed4, seg4).sum(dim=1, keepdim=True)

        # concat results from multiple resolutions
        results = [pred2, pred3, pred4]
        return [feats, results]
