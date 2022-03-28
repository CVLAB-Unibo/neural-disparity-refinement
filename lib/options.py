import argparse
from random import choices

class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):

        g_data = parser.add_argument_group("Data")
        g_data.add_argument(
            "--downsampling_factor",
            type=int,
            default=4,
            help="Factor used to resize RGB and Disparity in input.",
        )
        g_data.add_argument(
            "--upsampling_factor",
            type=float,
            default=1,
            help="Estimate final disparity using this factor."
                 "The final resolution will be: input_resolution / downsampling_factor * upsampling_factor.",
        )
        g_data.add_argument(
            "--disp_scale",
            type=float,
            default=1.,
            help="Scale disparity values by this factor before feeding them to the network."
                 "Useful for handling domain shift due to different disparity range w.r.t. the training domain."
                 "The final output will be adjusted at the original scale.",
        )
        g_data.add_argument(
            '--max_disp', 
            type=int, 
            default=256, 
            help="Max disparity value (num of classification bins)." 
                 "256 for sceneflow ckpt." 
                 "1024 for unreal4k."
        )
        g_data.add_argument(
            '--scale_factor16bit', 
            type=int, 
            default=256, 
            help="Scale Factor to decode 16bit disparity inputs"
        )
        # Inference Paths
        g_inference = parser.add_argument_group("Inference")
        g_inference.add_argument( 
            "--rgb", 
            type=str, 
            default=None, 
            help="path to the RGB image"
            )
        g_inference.add_argument( 
            "--disparity", 
            type=str, 
            default=None, 
            help="path to the noisy input disparity image"
        )
        g_inference.add_argument(
            "--load_checkpoint_path",
            type=str,
            default=None,
            help="path to checkpoint",
        )
        g_inference.add_argument(
            "--results_path", 
            type=str, 
            default="./results", 
            help="path to save results"
        )
        # Model related
        g_model = parser.add_argument_group("Model")
        g_model.add_argument(
            "--num_in_ch",
            type=int,
            default=4,
            choices=[1, 2, 3, 4],
            help="how many channels for CNN feature extractor",
        )
        g_model.add_argument(
            "--backbone",
            default="vgg13",
            help="backbone",
        )
        # special tasks
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        parser = None
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

    def parse(self):
        opt = self.gather_options()
        return opt
