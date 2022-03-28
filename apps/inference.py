import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from lib.evaluation_utils import predict
from lib.model import Refiner
from lib.options import BaseOptions
from lib.utils import pad_img, depad_img, img_loader, disp_loader
 
# get options
opt = BaseOptions().parse()

@torch.no_grad()
def test(opt):
    # set cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt.device = device

    # create net
    net = Refiner(opt).to(device=device)

    print("Using Network: ", net.name)

    def set_eval():
        net.eval()

    # load checkpoints
    if opt.load_checkpoint_path is None:
        raise ValueError("Missing path to checkpoint! Use --load_checkpoint_path flag")

    print("loading weights ...", opt.load_checkpoint_path)
    net.load_state_dict(
        torch.load(opt.load_checkpoint_path, map_location=device)["state_dict"]
    )

    set_eval()

    # create sample 
    rgb = img_loader(opt.rgb)
    rgb = cv2.resize(rgb, (rgb.shape[1] // opt.downsampling_factor, rgb.shape[0] // opt.downsampling_factor))
    height, width = rgb.shape[:2]
    
    disp = disp_loader(opt.disparity, opt.scale_factor16bit) / opt.disp_scale
    disp[disp > opt.max_disp] = 0
    height_disp, width_disp = disp.shape[:2]
    disp = np.expand_dims(cv2.resize(disp, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST),-1)
    rgb, pad = pad_img(rgb, height=height, width=width, divisor=32)
    disp, _ = pad_img(disp, height=height, width=width, divisor=32)

    rgb = torch.from_numpy(rgb).float()
    disp = torch.from_numpy(disp).float()
    o_shape = torch.from_numpy(np.asarray((height, width)))

    rgb = rgb.permute(2, 0, 1)
    disp = disp.permute(2, 0, 1)

    sample = {
        "rgb": rgb,
        "disp": disp,
        "o_shape": o_shape,
        "pad": pad,
    }

    pred, confidence = predict(
        net,
        device,
        sample,
        upscale_factor=opt.upsampling_factor,
    )

    # We scale disparity values accordingly to the final output resolution.
    pred = pred * (pred.shape[1] / width_disp)
    pred = pred * opt.disp_scale
    
    os.makedirs("%s" % (os.path.dirname(opt.results_path)), exist_ok=True)

    np.save(opt.results_path +  ".npy", pred)
    np.save(opt.results_path +  "_confidence.npy", confidence)

    plt.imsave(
        opt.results_path +  ".png",
        pred,
        cmap="magma"
    )

    plt.imsave(
        opt.results_path +  "_confidence.png",
        confidence,
        cmap="magma",
    )

if __name__ == "__main__":
    test(opt)
