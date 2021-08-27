from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import glob
import os


class CelebADataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=512)
        else:
            parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(display_winsize=512)
        parser.set_defaults(label_nc=18)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        parser.set_defaults(no_instance=True)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'val' if opt.phase == 'test' else 'train'

        # all_images = make_dataset(root, recursive=True, read_cache=False, write_cache=False)
        # image_paths = glob.glob(os.path.join(root, 'CelebA-HQ-img', '*.jpg'))
        # label_paths = glob.glob(os.path.join(root, 'CelebAMaskHQ-mask', '*.png'))

        if phase == 'val':
            filter_file = 'test_list.txt'
        else:
            filter_file = 'train_list.txt'

        image_filters = open(os.path.join(root, filter_file), 'rt').read().split('\n')
        image_filters = [x.split('.')[0] for x in image_filters]
        image_filters = dict.fromkeys(image_filters, None)

        image_paths = [os.path.join(root, 'CelebA-HQ-img', x + '.jpg') for x in image_filters if x.strip() != '']
        label_paths = [os.path.join(root, 'CelebAMaskHQ-mask', x + '.png') for x in image_filters if x.strip() != '']

        assert len(image_paths) == len(label_paths)
        for i in range(len(image_paths)):
            assert os.path.exists(image_paths[i])
            assert os.path.exists(label_paths[i])

        # for p in all_images:
        #     if '_%s_' % phase not in p:
        #         continue
        #     if p.endswith('.jpg'):
        #         image_paths.append(p)
        #     elif p.endswith('.png'):
        #         label_paths.append(p)

        instance_paths = []  # don't use instance map for celeba

        return label_paths, image_paths, instance_paths

    def postprocess(self, input_dict):
        label = input_dict['label']
        label = label - 1
        label[label == -1] = self.opt.label_nc
        input_dict['label'] = label.clamp(0, self.opt.label_nc)
