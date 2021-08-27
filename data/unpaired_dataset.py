from data.pix2pix_dataset import Pix2pixDataset
import glob
import os
from data.base_dataset import get_params, get_transform
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import os


class UnpairedDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=3)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        parser.set_defaults(no_instance=True)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'test' if opt.phase == 'test' else 'train'
        # print(os.path.join(root, f'{phase}A'))

        label_paths = glob.glob(os.path.join(root, f'{phase}A', '*.jpg'))
        image_paths = glob.glob(os.path.join(root, f'{phase}B', '*.jpg'))

        self.A_size = len(label_paths)
        self.B_size = len(image_paths)

        # print(self.A_size, self.B_size)
        # print(label_paths[0], image_paths[0])

        for i in range(len(image_paths)):
            assert os.path.exists(image_paths[i])

        for i in range(len(label_paths)):
            assert os.path.exists(label_paths[i])

        instance_paths = []  # don't use instance map for celeba

        return label_paths, image_paths, instance_paths

    def __getitem__(self, idx):
        # label image
        A_path = self.label_paths[idx % self.A_size]
        A_image = Image.open(A_path).convert('RGB')

        params = get_params(self.opt, A_image.size)
        transform_image = get_transform(self.opt, params)
        A_image_tensor = transform_image(A_image)

        # input_image (real images)
        B_path = self.image_paths[idx % self.B_size]
        B_image = Image.open(B_path).convert('RGB')
        B_image_tensor = transform_image(B_image)

        # # if using instance maps
        # if self.opt.no_instance:
        #     instance_tensor = 0
        # else:
        #     instance_path = self.instance_paths[idx]
        #     instance = Image.open(instance_path)
        #     if instance.mode == 'L':
        #         instance_tensor = transform_label(instance) * 255
        #         instance_tensor = instance_tensor.long()
        #     else:
        #         instance_tensor = transform_label(instance)

        input_dict = {'A': A_image_tensor, 'B': B_image_tensor, 'A_path': A_path, 'B_path': B_path}

        return input_dict

    def __len__(self):
        return max(self.A_size, self.B_size)
