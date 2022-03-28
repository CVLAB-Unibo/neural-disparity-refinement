import os
import re
import sys

import cv2
import numpy as np
import torch



def scale_coords(points, max_length):
    return 2 * points / (max_length - 1.0) - 1.0


def to_numpy(tensor):
    return tensor.squeeze().detach().cpu().numpy()


def interpolate(feat, uv):
    uv = uv.transpose(1, 2)
    uv = uv.unsqueeze(2)
    samples = torch.nn.functional.grid_sample(feat, uv)
    return samples[:, :, :, 0]


def load_ckp(checkpoint_path, cuda, model, optimizer):
    checkpoint = torch.load(checkpoint_path, map_location=cuda)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr = checkpoint["learning_rate"]
    return model, optimizer, checkpoint["epoch"], lr


def save_ckp(state, checkpoint_path):
    torch.save(state, checkpoint_path)


def readPFM(file):
    file = open(file, "rb")
    header = file.readline().rstrip()

    if (sys.version[0]) == "3":
        header = header.decode("utf-8")
    if header == "PF":
        color = True
    elif header == "Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    if (sys.version[0]) == "3":
        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
    else:
        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    if (sys.version[0]) == "3":
        scale = float(file.readline().rstrip().decode("utf-8"))
    else:
        scale = float(file.readline().rstrip())

    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def disp_loader(path, scale_factor16bit=256):
    disp = None
    if not os.path.exists(path):
        raise ValueError("Cannot open disp: " + path)

    if path.endswith("pfm"):
        disp = np.expand_dims(readPFM(path), 0)
    if path.endswith("png"):
        disp = np.expand_dims(cv2.imread(path, -1), 0)
        if disp.dtype == np.uint16:
            disp = disp / float(scale_factor16bit)
    if path.endswith("npy"):
        disp = np.expand_dims(np.load(path, mmap_mode="c"), 0)
    if disp is None:
        raise ValueError("Problems while loading the disp")
    # Remove invalid values
    disp[np.isinf(disp)] = 0

    return disp.transpose(1, 2, 0).astype(np.float32)

def resize_imgs(imgs):
    dim = len(imgs[0].shape)
    if dim==3:
        assert(imgs[0].shape[0]==3)
        height = imgs[0].shape[1]
        width = imgs[0].shape[2]
    elif dim==2:
        height = imgs[0].shape[0]
        width = imgs[0].shape[1]
    else:
        raise RuntimeError("Unsupported dimension!")

    for idx in range(1, len(imgs)):
        if dim==3:
            imgs[idx] = cv2.resize(imgs[idx].transpose(1,2,0), (width, height)).transpose(2,0,1)
        else:
            imgs[idx] = cv2.resize(imgs[idx], (width, height), interpolation = cv2.INTER_NEAREST)
    return
    
def img_loader(path, mode="passive", height=2160, width=3840):
    img = None
    if not os.path.exists(path):
        raise ValueError(f"Cannot open image: {path}")
    if path.endswith("raw"):
        img = (
            np.fromfile(open(path, "rb"), dtype=np.uint8).reshape(height, width, 3)
            if mode == "passive"
            else np.fromfile(open(path, "rb"), dtype=np.uint8).reshape(height, width, 1)
        )
    else:
        img = cv2.imread(path,1)
        if img.ndim == 2:
            img = np.expand_dims(img, -1) #

    return img


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    return lr


def shift_2d_replace(data, dx, dy, constant=False):
    shifted_data = np.roll(data, dx, axis=1)
    if dx < 0:
        shifted_data[:, dx:] = constant
    elif dx > 0:
        shifted_data[:, 0:dx] = constant

    shifted_data = np.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = constant
    elif dy > 0:
        shifted_data[0:dy, :] = constant
    return shifted_data


def rgb_loader(path, height=2160, width=3840, channels=3):
    img = None
    try:
        if path.endswith("raw"):
            img = np.fromfile(open(path, "rb"), dtype=np.uint8).reshape(
                height, width, channels
            )
        else:
            img = cv2.imread(path, -1)
    except:
        print("Cannot open RGB image: " + path)

    return img


def pad_img(img: np.ndarray, height: int = 1024, width: int = 1024, divisor: int = 32):
    """Pad the input image, making it larger at least (:attr:`height`, :attr:`width`)

    Params:
    ----------

    img (np.ndarray):
        array with shape h x w x c

    height (int):
        new minimum height

    width (int):
        new minimum width

    divisor (int):
        divisor factor, it forces the padded array to be multiple of divisor

    Returns:
        a new array with shape  H x W x c, multiple of divisior, and
        the amount of padding
    """
    h_pad = 0 if (height % divisor) == 0 else divisor - (height % divisor)
    top = h_pad // 2
    bottom = h_pad - top
    w_pad = 0 if (width % divisor) == 0 else divisor - (width % divisor)
    left = w_pad // 2
    right = w_pad - left
    img = np.lib.pad(img, ((top, bottom), (left, right), (0, 0)), mode="reflect")
    pad = np.stack([top, bottom, left, right], axis=0)
    return img, pad


def depad_img(
    img: np.ndarray,
    pad: np.ndarray,
    upsampling_factor: float = 1,
):
    """Remove padding from tensor

    Params:
    -------------

    img (np.ndarray):
        array to de-pad, with shape CxHxW or HxW
    pad (np.ndarray):
        array (top_pad, bottom_pad, left_pad, right_pad) with shape 1x4
    upsampling_factor (int):
        how to scale crops. For instance, if :attr:`upsampling_factor: is 4,
        crops are upscaled by 4.
        Default is 1.
    Returns:
    ------------
        a np.ndarray
    """

    if not img.ndim == 3:
        img = np.expand_dims(img, 0)
    pad = pad.squeeze()
    top = int(pad[0] * upsampling_factor)
    bottom = int(pad[1] * upsampling_factor)
    left = int(pad[2] * upsampling_factor)
    right = int(pad[3] * upsampling_factor)

    return img[
        :,
        top : img.shape[1] - bottom,
        left : img.shape[2] - right,
    ]

def get_boundaries(disp, th=1.0, dilation=10):
    edges_y = np.logical_or(
        np.pad(np.abs(disp[1:, :] - disp[:-1, :]) > th, ((1, 0), (0, 0))),
        np.pad(np.abs(disp[:-1, :] - disp[1:, :]) > th, ((0, 1), (0, 0))),
    )
    edges_x = np.logical_or(
        np.pad(np.abs(disp[:, 1:] - disp[:, :-1]) > th, ((0, 0), (1, 0))),
        np.pad(np.abs(disp[:, :-1] - disp[:, 1:]) > th, ((0, 0), (0, 1))),
    )
    edges = np.logical_or(edges_y, edges_x).astype(np.float32)

    if dilation > 0:
        kernel = np.ones((dilation, dilation), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    return edges