import math
import numpy as np
import torch
from lib.utils import to_numpy, get_boundaries, shift_2d_replace

def SEE(pred, gt, radius=2):
    abs_diff=[]
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            abs_diff.append(np.abs(shift_2d_replace(gt, i, j, 0) - pred))
    return np.minimum.reduce(abs_diff)

def compute_bad(disp_diff, gt, mask, th=3):
    bad_pixels =  disp_diff > th
    return 100.0 * bad_pixels.sum() / mask.sum()

def eval_disp(pred, gt, mask):
    epe = 0
    bad_2= 0
    bad_3= 0
    bad_4= 0
    bad_5= 0

    if mask.sum() > 0:
        disp_diff = np.abs(gt[mask] - pred[mask])
        bad_2 = compute_bad(disp_diff, gt, mask, 2.)
        bad_3 = compute_bad(disp_diff, gt, mask, 3.)
        bad_4 = compute_bad(disp_diff, gt, mask, 4.)
        bad_5 = compute_bad(disp_diff, gt, mask, 5.)
        epe = disp_diff.mean()
    return epe, bad_2, bad_3, bad_4, bad_5

def eval_edges(gt, pred, th=1., dilation=0):
    see_disp = 0
    edges = get_boundaries(gt, th=th, dilation=dilation)
    mask = gt > 0
    mask = np.logical_and(mask, edges)

    if mask.sum() > 0:
        see_disp = SEE(pred, gt)[mask].mean()

    return see_disp

def rmse(gt, pred):
    mask = gt > 0
    rmse = (gt[mask] - pred[mask]) ** 2
    return np.sqrt(rmse.mean())


def predict(net, cuda, data, upscale_factor=1, num_samples=50000):
    disp_tensor = data["disp"].to(device=cuda)
    img_tensor = data["rgb"].to(device=cuda)

    if img_tensor.ndim == 3:
        img_tensor = torch.unsqueeze(img_tensor, 0)
    if disp_tensor.ndim == 3:
        disp_tensor = torch.unsqueeze(disp_tensor, 0)

    x = torch.cat([img_tensor, disp_tensor], dim=1)

    input_height = data["o_shape"][0]
    input_width = data["o_shape"][1]
    output_height = data["o_shape"][0] * upscale_factor
    output_width = data["o_shape"][1] * upscale_factor

    net.filter(x)

    pad = data["pad"]
    res = inference(
        net,
        cuda,
        start_x=pad[2]*upscale_factor,
        end_x=(x[0].shape[2] - pad[3])*upscale_factor,
        start_y=pad[0]*upscale_factor,
        end_y=(x[0].shape[1] - pad[1])*upscale_factor,
        height=output_height,
        width=output_width,
        num_out=2,
        num_samples=num_samples,
    )
    pred = to_numpy(res[0])
    confidence = to_numpy(res[1])
    return pred, confidence


def inference(net, cuda, start_x=0, end_x=2048, start_y=0, end_y=1536, height=1536, width=2048, num_samples=200000, num_out=2):
    height = int(height)
    width = int(width)
    nx = np.linspace(start_x, end_x, width)
    ny = np.linspace(start_y, end_y, height)
    u, v = np.meshgrid(nx, ny)

    coords = np.expand_dims(np.stack((u.flatten(), v.flatten()), axis=-1), 0)
    batch_size, n_pts, _ = coords.shape
    coords = torch.Tensor(coords).float().to(device=cuda)
    output = torch.zeros(num_out, math.ceil(width * height / num_samples), num_samples)

    with torch.no_grad():
        for i, p_split in enumerate(
            torch.split(
                coords.reshape(batch_size, -1, 2), int(num_samples / batch_size), dim=1
            )
        ):
            points = torch.transpose(p_split, 1, 2)
            net.query(points.to(device=cuda))
            preds = net.get_disparity()
            confidence = net.get_confidence()
            output[0, i, : p_split.shape[1]] = preds.to(device=cuda)
            output[1, i, : p_split.shape[1]] = confidence.to(device=cuda)
    res = []
    for i in range(num_out):
        res.append(output[i].view(1, -1)[:, :n_pts].reshape(-1, height, width))
    return res