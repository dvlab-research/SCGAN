import sys
from collections import OrderedDict
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainer.pix2pix_trainer import Pix2PixTrainer
from option.train_options import TrainOptions
import yaml
import os
import torch
from tqdm import tqdm
from copy import deepcopy
from util import html
from util.fid import calculate_fid_given_paths


def load_opt(opt):
    if opt.config != '':
        assert (os.path.isfile(opt.config))
        opt_more = yaml.load(open(opt.config, 'r').read())

        for k in opt_more.keys():
            setattr(opt, k, opt_more[k])
    return opt


if __name__ == '__main__':
    # parse options
    opt = TrainOptions().parse()

    opt = load_opt(opt)

    print(' '.join(sys.argv))

    # load the dataset
    dataloader = data.create_dataloader(opt)

    testdataloader = None
    opt_test = deepcopy(opt)
    opt_test.results_dir = './results'
    opt_test.preprocess_mode = 'scale_width_and_crop'
    opt_test.serial_batches = True
    opt_test.no_flip = True
    opt_test.phase = 'test'
    opt_test.how_many = float('inf')

    fid_best = float('inf')

    # create trainer for our model
    trainer = Pix2PixTrainer(opt)

    # create tool for counting iterations
    iter_counter = IterationCounter(opt, len(dataloader))

    # create tool for visualization
    visualizer = Visualizer(opt)

    for epoch in tqdm(iter_counter.training_epochs()):
    # for epoch in iter_counter.training_epochs():
        iter_counter.record_epoch_start(epoch)
        for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()

            # training
            # train generator
            if i % opt.D_steps_per_G == 0:
                trainer.run_generator_one_step(data_i)

            # train discriminator
            trainer.run_discriminator_one_step(data_i)
            # visualizations
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter, losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

            if iter_counter.needs_displaying():
                visuals = OrderedDict([('input_label', data_i['label'][:opt.visual_n, :, :, :]),
                                       ('synthesized_image', trainer.get_latest_generated()[:opt.visual_n, :, :, :]),
                                       ('real_image', data_i['image'][:opt.visual_n, :, :, :]),
                                       ('w_image', trainer.w_image[:opt.visual_n, :, :, :])])
                visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

            if iter_counter.needs_saving():
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()

            del data_i

        trainer.update_learning_rate(epoch)
        iter_counter.record_epoch_end()

        if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            trainer.save(epoch)

        if epoch % opt.fid_epoch == 0 or epoch == iter_counter.total_epochs:
            # fid evaluation
            torch.cuda.empty_cache()

            opt_test.which_epoch = str(epoch)

            if testdataloader is None:
                testdataloader = data.create_dataloader(opt_test)

            model = trainer.pix2pix_model
            model.eval()

            visualizer = Visualizer(opt_test)

            # create a webpage that summarizes the all results
            web_dir = os.path.join(opt_test.results_dir, opt.name, '%s_%s' % (opt_test.phase, opt_test.which_epoch))
            webpage = html.HTML(web_dir,
                                'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name,
                                                                             opt_test.phase,
                                                                             opt_test.which_epoch))

            preds, gts = [], []

            for j, data_j in enumerate(testdataloader):
                if j * opt_test.batchSize >= opt_test.how_many:
                    break

                generated = model(data_j, mode='inference')

                img_path = data_j['path']
                for b in range(generated.shape[0]):
                    print('process image... %s' % img_path[b])
                    visuals = OrderedDict([('input_label', data_j['label'][b]),
                                        ('synthesized_image', generated[b])])
                    visualizer.save_images(webpage, visuals, img_path[b:b+1])

            webpage.save()

            fid_value = calculate_fid_given_paths([f'/data/datasets/syn-gts/{opt_test.dataset}', f'results/{opt_test.name}/test_{opt_test.which_epoch}/images/synthesized_image/'], batch_size=40, cuda=True, dims=2048)

            with open(f'{opt.name}.txt', 'a') as fr:
                fr.write(f'{opt_test.which_epoch}, fid: {str(fid_value)}')

            if fid_value < fid_best:
                fid_best = fid_value
                trainer.save('best')
                print(f'current best fid: {fid_best}')

            trainer.pix2pix_model.train()
            torch.cuda.empty_cache()

    print('Training was successfully finished.')