from model.sync_batchnorm import DataParallelWithCallback
from model.pix2pix_model import Pix2PixModel
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import pdb
from ptflops import get_model_complexity_info

class Pix2PixTrainer():
    """
    Trainer creates the model and optimizers, and uses them to updates
    the weights of the network while reporting losses and the latest
    visuals to visualize the progress in training
    """
    def __init__(self, opt):
        self.opt = opt
        self.pix2pix_model = Pix2PixModel(opt)
        # self.pix2pix_model_on_one_gpu = self.pix2pix_model
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = self.pix2pix_model.create_optimizers(opt)
            self.old_lr = opt.lr
        if len(opt.gpu_ids) > 0:
            # self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model, device_ids=opt.gpu_ids)
            if opt.multi_thread_gpu:
                local_rank = torch.distributed.get_rank()
                # torch.cuda.set_device(local_rank)
                # device = torch.device('cuda', local_rank)
                # self.pix2pix_model = self.pix2pix_model
                self.pix2pix_model = DDP(self.pix2pix_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
                # self.pix2pix_model = DDP(self.pix2pix_model, device_ids=[local_rank], output_device=local_rank)
                # self.pix2pix_model.cuda(opt.gpu_ids[0])
                # self.pix2pix_model = DDP(self.pix2pix_model)
            else:
                # self.pix2pix_model = torch.nn.DataParallel(self.pix2pix_model, device_ids=opt.gpu_ids)
                self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model, device_ids=opt.gpu_ids)
            # os.environ['MASTER_ADDR'] = 'localhost'
            # os.environ['MASTER_PORT'] = '12345'
            # dist.init_process_group("p2p", world_size=opt.nThreads)
            # torch.manual_seed(42)
            # self.pix2pix_model = DDP(self.pix2pix_model, device_ids=opt.gpu_ids)

            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
        else:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model

        self.generated = None
        self.w_image = None
        self.routing_weights = None
        # if opt.isTrain:
        #     self.optimizer_G, self.optimizer_D = self.pix2pix_model_on_one_gpu.create_optimizers(opt)
        #     self.old_lr = opt.lr

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        # print(data['image'].size())
        # macs, params = get_model_complexity_info(self.pix2pix_model, (3, 224, 224), input_constructor=data, as_strings=True,
        #                                          print_per_layer_stat=True, verbose=True)
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        g_losses, generated, w_image, _ = self.pix2pix_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated
        self.w_image = w_image

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        d_losses = self.pix2pix_model(data, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def update_learning_rate(self, epoch): # ??? endless recursion
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr_D = new_lr
            else:
                new_lr_G, new_lr_D = new_lr/2, new_lr*2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D

            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G

            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)
