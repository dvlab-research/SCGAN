from data.pix2pix_dataset import Pix2pixDataset
import glob
import os
from data.base_dataset import get_params, get_transform
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import os


class ISDataset(Pix2pixDataset):

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

        print(os.path.join(root))

        image_paths = glob.glob(os.path.join(root, '*.png'))
        self.size = len(image_paths)

        print(self.size)

        # print(self.A_size, self.B_size)
        # print(label_paths[0], image_paths[0])

        for i in range(len(image_paths)):
            assert os.path.exists(image_paths[i])

        label_paths = []
        instance_paths = []

        return label_paths, image_paths, instance_paths

    def __getitem__(self, idx):

        # input_image (real images)
        B_path = self.image_paths[idx]
        B_image = Image.open(B_path).convert('RGB')

        params = get_params(self.opt, B_image.size)
        transform_image = get_transform(self.opt, params)

        B_image_tensor = transform_image(B_image)

        return B_image_tensor

    def __len__(self):
        return self.size
