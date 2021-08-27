import os
from collections import OrderedDict

import data
import numpy as np
from option.test_options import TestOptions
from model.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
import yaml

opt = TestOptions().parse()

if opt.config != '':
    assert(os.path.isfile(opt.config))
    opt_more = yaml.load(open(opt.config, 'r').read())

    for k in opt_more.keys():
        setattr(opt, k, opt_more[k])

dataloader = data.create_dataloader(opt)
print(opt.dataset)

# opt.batchSize = 1
model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test

preds, gts = [], []

for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    if os.path.exists(data_i['path'][0]) and opt.no_overwrite:
        continue

    generated = model(data_i, mode='inference')

    if opt.fid is True:
        preds.append(generated.cpu().detach().numpy())
        gts.append(data_i['image'].cpu().detach().numpy())

    if opt.no_record is False:
        img_path = data_i['path']
        for b in range(generated.shape[0]):
            print('process image... %s' % img_path[b])
            visuals = OrderedDict([('input_label', data_i['label'][b]),
                                   ('synthesized_image', generated[b])])
            visualizer.save_images(webpage, visuals, img_path[b:b+1])

webpage.save()

dataset = opt.dataset

if opt.fid is True:

    os.system(f'python fid.py /data/datasets/syn-gts/{opt.dataset} results/{opt.name}/test_{opt.which_epoch}/'
              f'images/synthesized_image/ --batch-size 40')