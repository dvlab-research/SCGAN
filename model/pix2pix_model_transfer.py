import torch
import torch.nn as nn
import util.func as util
import model.net as networks
from model.ops.loss import GANLoss, VGGLoss, KLDLoss, DiversityLoss
from model.ops.local import Erode
import numpy as np
from scipy.stats import truncnorm


class Pix2PixModel(nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() else torch.ByteTensor

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = nn.L1Loss()
            self.criterionSeg = nn.CrossEntropyLoss(ignore_index=255)
            self.criterionRouting = DiversityLoss()
            if not opt.no_vgg_loss:
                self.criterionVGG = VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = KLDLoss()

        # partially sampling image contents
        self.nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label else self.opt.label_nc
        # self.sampling_classes = (self.FloatTensor(self.nc).zero_()+1) * self.opt.cls_prob
        # self.coin = self.FloatTensor(1).zero_()

        if opt.erode:
            self.erode = Erode(iters=2).cuda()

        if opt.texturize:
            # self.noise_maps = self.FloatTensor(1, self.nc, 256, 256).zero_()
            std_gap = 10.0 / self.nc
            self.noise_stds = [std_gap * i for i in range(1, self.nc+1)]

        if opt.use_truncnorm:
            print('use truncnorm')

    def forward(self, data, mode='generator'):

        if mode == 'generator':
            input_semantics, real_image = self.preprocess_input(data)
            g_loss, generated, w_image, routing_weights = self.compute_generator_loss(input_semantics, real_image)
            return g_loss, generated, w_image, routing_weights
        elif mode == 'discriminator':
            input_semantics, real_image = self.preprocess_input(data)
            d_loss = self.compute_discriminator_loss(input_semantics, real_image)
            return d_loss
        elif mode == 'encode_only':
            input_semantics, real_image = self.preprocess_input(data)
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            input_semantics, real_image = self.preprocess_input(data)
            with torch.no_grad():
                fake_image, _, _, _ = self.generate_fake(input_semantics, real_image)
            return fake_image
        elif mode == 'unpair':
            with torch.no_grad():
                data_semantics = data[0]
                data_style = data[1]
                input_semanticsA, real_imageA = self.preprocess_input(data_semantics)
                input_semanticsB, real_imageB = self.preprocess_input(data_style)
                fake_image, w_image, _, _ = self.generate_fake(input_semanticsA, real_imageB)
            return fake_image, w_image
        else:
            raise ValueError('|mode| is invalid')

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr = D_lr = opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            print(f'load G at epoch {opt.which_epoch}')
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain and (not opt.dont_load_D):
                print('load D')
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                print('load E')
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)

        return netG, netD, netE

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['instance'] = data['instance'].cuda()
            data['image'] = data['image'].cuda()

        # create one-hot-label map
        label_map = data['label']
        bs, _, h, w = label_map.size()

        input_label = self.FloatTensor(bs, self.nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0) # one-hot vector
        if self.opt.erode:
            input_semantics = self.erode(input_semantics)

        if self.opt.texturize:
            noise_maps = torch.zeros(1, self.nc, 256, 256).cuda()
            for i in range(self.nc):
                noise_maps[:, i, :, :].data.normal_(0, self.noise_stds[i])

            input_semantics = input_semantics * noise_maps
            del noise_maps

        if self.opt.sampling !='none':
            coin = torch.zeros(1).cuda()
            if coin.uniform_() <= self.opt.sampling_prob: # let us sample a portion of input
                sampling_classes = torch.ones(self.nc).cuda() * self.opt.cls_prob
                if self.opt.sampling == 'uniform':
                    sampling_cls = (sampling_classes.uniform_() <= self.opt.cls_prob).view(1, self.nc, 1, 1).float()
                else: # multinomial
                    sampling_cls_indicator = self.FloatTensor(1, self.nc, 1, 1)
                    sampling_cls = sampling_classes.multinomial_(self.nc // 10)
                    sampling_cls = sampling_cls_indicator.scatter_(1, sampling_cls, 1.0) # one-hot vector
                input_semantics = input_semantics * sampling_cls

        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)
        return input_semantics, data['image']

    def compute_generator_loss(self, input_semantics, real_image):
        G_losses = {}
        fake_image, w_image, routing_weights, KLD_loss = self.generate_fake(input_semantics, real_image,
                                                                            compute_kld_loss=self.opt.use_vae)

        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        if 'fpse' in self.opt.netD:
            # print('discriminate')
            feat_fake, pred_fake, feat_real, pred_real = self.discriminate(
                input_semantics, fake_image, real_image)
            # print('after discriminate')
            if not self.opt.no_ganFeat_loss:
                GAN_Feat_loss = self.FloatTensor(1).fill_(0)
                num_D = len(feat_fake)
                for i in range(num_D):
                    GAN_Feat_loss += self.criterionFeat(
                        feat_fake[i], feat_real[i].detach()) * self.opt.lambda_feat / num_D
                G_losses['GAN_Feat'] = GAN_Feat_loss
        elif 'routing' in self.opt.netD:
            feat_fake, pred_fake, feat_real, pred_real = self.discriminate(
                input_semantics, fake_image, real_image, routing_weights)
            # print('after discriminate')
            if not self.opt.no_ganFeat_loss:
                GAN_Feat_loss = self.FloatTensor(1).fill_(0)
                num_D = len(feat_fake)
                for i in range(num_D):
                    GAN_Feat_loss += self.criterionFeat(
                        feat_fake[i], feat_real[i].detach()) * self.opt.lambda_feat / num_D
                G_losses['GAN_Feat'] = GAN_Feat_loss
        else:
            pred_fake, pred_real = self.discriminate(input_semantics, fake_image, real_image)

            if not self.opt.no_ganFeat_loss:
                num_D = len(pred_fake)
                GAN_Feat_loss = self.FloatTensor(1).fill_(0)
                for i in range(num_D): # for each discriminator
                    num_intermediate_outputs = len(pred_fake[i]) - 1
                    for j in range(num_intermediate_outputs): # for each layer output
                        unweighted_loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                        GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
                G_losses['GAN_Feat'] = GAN_Feat_loss

        G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)
        # print('GAN')
        # if 'labelenc' in self.opt.netG:
        #     G_losses['W_Recon'] = self.criterionSeg(w_image, self.input_label_buf) * self.opt.lambda_w_recon
        #     w_image = torch.argmax(w_image, dim=1, keepdim=True)
        # else:
        if self.opt.diversity_loss:
            G_losses['W_Recon'] = self.criterionRouting(w_image, real_image.detach(),
                                                        input_semantics.detach()) * self.opt.lambda_w_recon
            w_image = w_image[::9, :, :, :]
        else:
            G_losses['W_Recon'] = self.criterionFeat(w_image, real_image) * self.opt.lambda_w_recon

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg
        # print('vgg')
        return G_losses, fake_image, w_image, routing_weights

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image, _, routing_weights, _ = self.generate_fake(input_semantics, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        if 'fpse' in self.opt.netD:
            _, pred_fake, _, pred_real = self.discriminate(input_semantics, fake_image, real_image)
        elif 'routing' in self.opt.netD:
            _, pred_fake, _, pred_real = self.discriminate(input_semantics, fake_image, real_image, routing_weights)
        else:
            pred_fake, pred_real = self.discriminate(input_semantics, fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True, for_discriminator=True)
        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False):
        z, KLD_loss = None, None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        fake_image, w_image, routing_weights = self.netG(input_semantics, z)

        assert (not compute_kld_loss) or self.opt.use_vae, 'You cannot compute KLD loss if opt.use_vae == False'

        return fake_image, w_image, routing_weights, KLD_loss

    def discriminate(self, input_semantics, fake_image, real_image, routing_weights=None):
        # print('real_image shape: ', real_image.shape)
        if 'fpse' in self.opt.netD:
            fake_and_real_img = torch.cat([fake_image, real_image], dim=0)

            discriminator_out = self.netD(fake_and_real_img,
                                          segmap=torch.cat((input_semantics, input_semantics), dim=0))

            fake_feats, fake_preds, real_feats, real_preds = self.divide_pred(discriminator_out)

            return fake_feats, fake_preds, real_feats, real_preds
        elif 'routing' in self.opt.netD:
            fake_and_real_img = torch.cat([fake_image, real_image], dim=0)

            discriminator_out = self.netD(fake_and_real_img,
                                          segmap=torch.cat((input_semantics, input_semantics), dim=0),
                                          routing_weights=routing_weights)

            fake_feats, fake_preds, real_feats, real_preds = self.divide_pred(discriminator_out)

            return fake_feats, fake_preds, real_feats, real_preds
        else:
            fake_concat = torch.cat([input_semantics, fake_image], dim=1)
            real_concat = torch.cat([input_semantics, real_image], dim=1)
            # print('fake vs real: ', fake_concat.shape, real_concat.shape)
            fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

            discriminator_out = self.netD(fake_and_real)

            pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN
        # so it is usually a list
        if 'fpse' in self.opt.netD or 'routing' in self.opt.netD:
            fake_feats = []
            fake_preds = []
            real_feats = []
            real_preds = []
            for p in pred[0]:
                fake_feats.append(p[:p.size(0) // 2])
                real_feats.append(p[p.size(0) // 2:])
            for p in pred[1]:
                fake_preds.append(p[:p.size(0) // 2])
                real_preds.append(p[p.size(0) // 2:])

            return fake_feats, fake_preds, real_feats, real_preds
        else:
            if type(pred) == list:
                fake = []
                real = []
                for p in pred:
                    fake.append([tensor[:tensor.size(0)//2] for tensor in p])
                    real.append([tensor[tensor.size(0)//2:] for tensor in p])
            else:
                fake = pred[:pred.size(0)//2]
                real = pred[pred.size(0)//2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)

        if self.opt.use_truncnorm:
            b, c = std.shape[:2]
            eps = torch.from_numpy(truncnorm.rvs(-self.opt.tn_rg,
                                                 self.opt.tn_rg, size=(b, c)).astype(np.float32)).float().cuda()
        else:
            eps = torch.randn_like(std)
        # eps.normal_(0.0, self.opt.truncnorm_std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
