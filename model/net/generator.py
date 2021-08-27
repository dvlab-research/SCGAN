from functools import partial
from easydict import EasyDict as edict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from model.net.basenet import BaseNetwork
from model.ops.layer import IterativeGaussian
from model.ops.layer import ResnetBlock, SPADEResnetBlock, SCResnetBlock
from model.ops.norm import get_nonspade_norm_layer
from model.ops.norm import SpatialCondNorm, SPADE, SynchronizedBatchNorm2d


class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # filter number
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            self.fc = nn.Linear(opt.z_dim, 16*nf*self.sw*self.sh)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16*nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16*nf, 16*nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16*nf, 16*nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16*nf, 16*nf, opt)

        self.up_0 = SPADEResnetBlock(16*nf, 8*nf, opt)
        self.up_1 = SPADEResnetBlock(8*nf, 4*nf, opt)
        self.up_2 = SPADEResnetBlock(4*nf, 2*nf, opt)
        self.up_3 = SPADEResnetBlock(2*nf, 1*nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1*nf, nf//2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' % opt.num_upsampling_layers)
        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)
        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=seg.get_device())
            x = self.fc(z).view(-1, 16*self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.opt.num_upsampling_layers == 'more' or self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x


class Pix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4,
                            help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9,
                            help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super(Pix2PixHDGenerator, self).__init__()
        input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        actvation = nn.ReLU(False)

        model = []

        # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  actvation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf*mult, opt.ngf*mult*2,
                                           kernel_size=3, stride=2, padding=1)),
                      actvation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf*mult, norm_layer=norm_layer, actvation=actvation,
                                  kernel_size=opt.resnet_kernel_size)]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf*mult)
            nc_out = int((opt.ngf*mult)/2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out, kernel_size=3,
                                                    stride=2, padding=1, output_padding=1)),
                      actvation]
            mult //=2

        # final output conv
        model += [nn.ReflectionPad2d(3), nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        return self.model(input)


class SemanticVectorGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # filter number
        nf = opt.nrf
        input_nc = opt.semantic_nc
        self.n_conv = opt.num_experts_conv
        self.n_norm = opt.num_experts_norm

        scc_config_str = opt.norm_E.replace('spectral', '')

        if 'scc' in scc_config_str:
            self.norm_fn = SpatialCondNorm
        elif 'spade' in scc_config_str:
            self.norm_fn = SPADE
        else:
            if 'instance' in scc_config_str:
                self.norm_fn = partial(nn.InstanceNorm2d, affine=False)
            elif 'batch' in scc_config_str:
                self.norm_fn = partial(nn.BatchNorm2d, affine=True)
            elif 'sync_batch' in scc_config_str:
                self.norm_fn = partial(SynchronizedBatchNorm2d, affine=True)
            else:
                raise ValueError('normalization layer %s is not recognized' % scc_config_str)

        self.use_unet = opt.use_unet

        if self.opt.no_spectral_on_routing:
            self.labelDec1 = nn.Sequential(
                nn.Conv2d(in_channels=input_nc, out_channels=nf * 16, kernel_size=3, padding=1, stride=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_channels=nf * 16, out_channels=nf * 16, kernel_size=3, padding=1, stride=1),
            )

            self.labelDec2 = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_channels=nf * 16 + input_nc, out_channels=nf * 8, kernel_size=3, padding=1, stride=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_channels=nf * 8, out_channels=nf * 8, kernel_size=3, padding=1, stride=1),
            )

            self.labelDec3 = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_channels=nf * 8 + input_nc, out_channels=nf * 4, kernel_size=3, padding=1, stride=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_channels=nf * 4, out_channels=nf * 4, kernel_size=3, padding=1, stride=1),
            )

            self.labelDec4 = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_channels=nf * 4 + input_nc, out_channels=nf * 2, kernel_size=3, padding=1, stride=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_channels=nf * 2, out_channels=nf * 2, kernel_size=3, padding=1, stride=1),
            )

            self.labelDec5 = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_channels=nf * 2 + input_nc, out_channels=nf * 1, kernel_size=3, padding=1, stride=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, padding=1, stride=1),
            )

            self.labelDec6 = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_channels=nf + input_nc, out_channels=nf, kernel_size=3, padding=1, stride=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, padding=1, stride=1),
            )

            if opt.num_upsampling_layers == 'more':
                self.labelDec7 = nn.Sequential(
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(in_channels=nf + input_nc, out_channels=nf, kernel_size=3, padding=1, stride=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, padding=1, stride=1),
                )

            self.toRGB = nn.Sequential( # redudent
                nn.Conv2d(in_channels=nf, out_channels=nf // 2, kernel_size=3, padding=1, stride=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_channels=nf//2, out_channels=3, kernel_size=3, padding=1, stride=1),
                nn.Hardtanh()
            )
        else:
            self.labelDec1 = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels=input_nc, out_channels=nf * 16, kernel_size=3, padding=1, stride=1)),
                nn.LeakyReLU(0.2, True),
                spectral_norm(nn.Conv2d(in_channels=nf * 16, out_channels=nf * 16, kernel_size=3, padding=1, stride=1)),
            )

            self.labelDec2 = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                spectral_norm(nn.Conv2d(in_channels=nf * 16 + input_nc, out_channels=nf * 8, kernel_size=3, padding=1, stride=1)),
                nn.LeakyReLU(0.2, True),
                spectral_norm(nn.Conv2d(in_channels=nf * 8, out_channels=nf * 8, kernel_size=3, padding=1, stride=1)),
            )

            self.labelDec3 = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                spectral_norm(nn.Conv2d(in_channels=nf * 8 + input_nc, out_channels=nf * 4, kernel_size=3, padding=1, stride=1)),
                nn.LeakyReLU(0.2, True),
                spectral_norm(nn.Conv2d(in_channels=nf * 4, out_channels=nf * 4, kernel_size=3, padding=1, stride=1)),
            )

            self.labelDec4 = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                spectral_norm(nn.Conv2d(in_channels=nf * 4 + input_nc, out_channels=nf * 2, kernel_size=3, padding=1, stride=1)),
                nn.LeakyReLU(0.2, True),
                spectral_norm(nn.Conv2d(in_channels=nf * 2, out_channels=nf * 2, kernel_size=3, padding=1, stride=1)),
            )

            self.labelDec5 = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                spectral_norm(nn.Conv2d(in_channels=nf * 2 + input_nc, out_channels=nf * 1, kernel_size=3, padding=1, stride=1)),
                nn.LeakyReLU(0.2, True),
                spectral_norm(nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, padding=1, stride=1)),
            )

            self.labelDec6 = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                spectral_norm(nn.Conv2d(in_channels=nf + input_nc, out_channels=nf, kernel_size=3, padding=1, stride=1)),
                nn.LeakyReLU(0.2, True),
                spectral_norm(nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, padding=1, stride=1)),
            )

            if opt.num_upsampling_layers == 'more':
                self.labelDec7 = nn.Sequential(
                    nn.LeakyReLU(0.2, True),
                    spectral_norm(nn.Conv2d(in_channels=nf + input_nc, out_channels=nf, kernel_size=3, padding=1, stride=1)),
                    nn.LeakyReLU(0.2, True),
                    spectral_norm(nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, padding=1, stride=1)),
                )

            self.toRGB = nn.Sequential( # redudent
                nn.Conv2d(in_channels=nf, out_channels=nf // 2, kernel_size=3, padding=1, stride=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_channels=nf//2, out_channels=3, kernel_size=3, padding=1, stride=1),
                nn.Hardtanh()
            )

        self.act_fn = nn.Sigmoid() if opt.weight_type == 'sigmoid' else nn.Softmax2d()

        self.up = nn.Upsample(scale_factor=2)

        self.iter_gauss = IterativeGaussian(win_size=self.opt.ig_sz, iters=self.opt.ig_iter, sigma=self.opt.ig_sigma,
                                            relative=self.opt.ig_relative) if self.opt.position else None

        self.init_weights(opt.init_type, opt.init_variance)

    def enc2weight(self, x, ch):
        return F.adaptive_avg_pool3d(x.unsqueeze(1), (ch, x.size(2), x.size(3))).squeeze(1)

    def forward(self, label):
        layers = 6 if self.opt.num_upsampling_layers == 'normal' else 7
        muti_scale_labels = [0] * layers

        if self.iter_gauss and (not self.opt.pyr_pos):
            if not self.opt.no_instance:
                semantics, edges = self.iter_gauss(label[:, :-1, :, :]), label[:, -1:, :, :]
                label = torch.cat((semantics, edges), dim=1)
            else:
                label = self.iter_gauss(label)

        muti_scale_labels[layers-1] = label
        for i in range(layers-2, -1, -1):
            muti_scale_labels[i] = F.interpolate(muti_scale_labels[i+1], scale_factor=0.5, mode='nearest') if self.opt.pyr_pos else F.interpolate(muti_scale_labels[i+1], scale_factor=0.5, mode='bilinear')  # we used to use 'bilinear'
            if self.opt.pyr_pos and layers-i < self.opt.pyr_rng: # 2, 3, 4
                muti_scale_labels[i] = self.iter_gauss(muti_scale_labels[i])

        if self.opt.pyr_pos:
            muti_scale_labels[layers-1] = self.iter_gauss(muti_scale_labels[layers-1])

        if self.opt.labelnoise:
            for i in range(len(muti_scale_labels)):
                _label = muti_scale_labels[i]
                b, _, h, w = _label.size()
                _noise = torch.randn(b, 1, h, w).cuda()
                muti_scale_labels[i] = muti_scale_labels[i] * _noise

        x = self.labelDec1(muti_scale_labels[0])
        routing_w1 = self.enc2weight(x, self.n_conv)
        routing_w1_n = self.enc2weight(x, self.n_norm)
        x = self.up(x)

        x = self.labelDec2(torch.cat((x, muti_scale_labels[1]), dim=1))
        routing_w2 = self.enc2weight(x, self.n_conv)
        routing_w2_n = self.enc2weight(x, self.n_norm)
        x = self.up(x)

        x = self.labelDec3(torch.cat((x, muti_scale_labels[2]), dim=1))
        routing_w3 = self.enc2weight(x, self.n_conv)
        routing_w3_n = self.enc2weight(x, self.n_norm)
        x = self.up(x)

        x = self.labelDec4(torch.cat((x, muti_scale_labels[3]), dim=1))
        routing_w4 = self.enc2weight(x, self.n_conv)
        routing_w4_n = self.enc2weight(x, self.n_norm)
        x = self.up(x)

        x = self.labelDec5(torch.cat((x, muti_scale_labels[4]), dim=1))
        routing_w5 = self.enc2weight(x, self.n_conv)
        routing_w5_n = self.enc2weight(x, self.n_norm)
        x = self.up(x)

        x = self.labelDec6(torch.cat((x, muti_scale_labels[5]), dim=1))
        routing_w6 = self.enc2weight(x, self.n_conv)
        routing_w6_n = self.enc2weight(x, self.n_norm)

        if self.opt.num_upsampling_layers == 'more':
            x = self.up(x)
            x = self.labelDec7(torch.cat((x, muti_scale_labels[6]), dim=1))
            routing_w7 = self.enc2weight(x, self.n_conv)
            routing_w7_n = self.enc2weight(x, self.n_norm)

        im = self.toRGB(x)

        if self.opt.num_upsampling_layers == 'normal':
            ret = {'weights': [routing_w1, routing_w2, routing_w3, routing_w4, routing_w5, routing_w6], 'im': im,
                   'weights_norm': [routing_w1_n, routing_w2_n, routing_w3_n, routing_w4_n, routing_w5_n, routing_w6_n]}
        else:
            ret = {'weights': [routing_w1, routing_w2, routing_w3, routing_w4, routing_w5, routing_w6, routing_w7],
                   'im': im,
                   'weights_norm': [routing_w1_n, routing_w2_n, routing_w3_n, routing_w4_n, routing_w5_n, routing_w6_n,
                                    routing_w7_n]}
        return ret


class SCGGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # filter number
        nf = opt.ngf

        scc_config_str = opt.norm_G.replace('spectral', '')
        self.norm_config = scc_config_str

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            self.fc = nn.Linear(opt.z_dim, 16 * opt.nff * self.sw * self.sh)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * opt.nff, 3, padding=1)

        self.pyramid_spaital_routing = SemanticVectorGenerator(opt)

        self.head_0 = SCResnetBlock(16 * opt.nff, 16 * nf, opt)

        self.G_middle_0 = SCResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SCResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SCResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SCResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SCResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SCResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SCResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

        self.init_weights(opt.init_type, opt.init_variance)

        act_candidates = {'sigmoid': nn.Sigmoid, 'softmax': nn.Softmax2d, 'hardtanh': nn.Hardtanh}
        self.act_fn = act_candidates[opt.weight_type]() if opt.weight_type in act_candidates else lambda x: x

        self.t = self.opt.temperature

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' % opt.num_upsampling_layers)
        sw = opt.crop_size // (2 ** num_up_layers)
        sh = round(sw / opt.aspect_ratio)
        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if self.opt.use_vae:
            if z is None:
                z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=seg.get_device())
            x = self.fc(z).view(-1, 16 * self.opt.nff, self.sh, self.sw)
        else:
            if self.opt.model == 'pix2pixunpaired':
                x = F.interpolate(z, size=(self.sh, self.sw))
                x = self.fc(x)
            else:
                x = F.interpolate(seg, size=(self.sh, self.sw))
                x = self.fc(x)

        rd = self.pyramid_spaital_routing(seg)
        routing_weights_array = rd['weights']

        if self.opt.routing_detach:
            routing_weights_array = [x.detach() for x in routing_weights_array]

        rwa = rd['weights_norm']
        rimg = rd['im']

        x = self.head_0(x, (self.act_fn(routing_weights_array[0] * self.t), self.act_fn(rwa[0] * self.t)), seg)

        x = self.up(x)  # 16
        x = self.G_middle_0(x, (self.act_fn(routing_weights_array[1] * self.t), self.act_fn(rwa[1] * self.t)), seg)

        nult = 0
        if self.opt.num_upsampling_layers == 'more' or self.opt.num_upsampling_layers == 'most':
            nult = 1
            x = self.up(x)

        x = self.G_middle_1(x, (self.act_fn(routing_weights_array[1+nult] * self.t), self.act_fn(rwa[1+nult] * self.t)), seg)

        x = self.up(x)  # 32
        x = self.up_0(x, (self.act_fn(routing_weights_array[2+nult] * self.t), self.act_fn(rwa[2+nult] * self.t)), seg)

        x = self.up(x)  # 64
        x = self.up_1(x, (self.act_fn(routing_weights_array[3+nult] * self.t), self.act_fn(rwa[3+nult] * self.t)), seg)
        x = self.up(x)  # 128
        x = self.up_2(x, (self.act_fn(routing_weights_array[4+nult] * self.t), self.act_fn(rwa[4+nult] * self.t)), seg)
        x = self.up(x)  # 256

        x = self.up_3(x, (self.act_fn(routing_weights_array[5+nult] * self.t), self.act_fn(rwa[5+nult] * self.t)), seg)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.hardtanh(x)

        return x, rimg, routing_weights_array