from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import os
import util.func as func


class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='if specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt

        label_paths, image_paths, instance_paths = self.get_paths(self.opt)

        func.natural_sort(label_paths)
        func.natural_sort(image_paths)

        if not opt.no_instance:
            func.natural_sort(instance_paths)

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        instance_paths = instance_paths[:opt.max_dataset_size]

        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), 'The label-image pair (%s, %s) do not look liek the right pair'

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.instance_paths = instance_paths

        self.dataset_size = len(self.label_paths)

    def get_paths(self, opt): # it must be override
        label_paths, image_paths, instance_paths = [], [], []
        assert False, 'A subclass of Pix2pixDataset must override self.get_paths(self, opt)'
        return label_paths, image_paths, instance_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, idx):
        # label image
        label_path = self.label_paths[idx]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc # 'unknown' is opt.label_nc

        # input_image (real images)
        image_path = self.image_paths[idx]
        # assert self.paths_match(label_path, image_path), \
        #     'the label_path %s and image_path %s do not match.' % (label_path, image_path)
        image = Image.open(image_path).convert('RGB')

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[idx]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path}
        self.postprocess(input_dict)
        return input_dict

    def __len__(self):
        return self.dataset_size

    def postprocess(self, input_dict):
        return input_dict
