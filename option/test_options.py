from option.base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')
        parser.add_argument('--fid', action='store_true', help='using fid')
        parser.add_argument('--no_record', action='store_true', help='do not record results.')
        parser.add_argument('--use_unet', action='store_true', help='use unet to predict routing weights')
        parser.add_argument('--no_overwrite', action='store_true', help='do not overwrite the existing results')
        parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=256, load_size=256, display_winsize=256)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        self.isTrain = False
        return parser
