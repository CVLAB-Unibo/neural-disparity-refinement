import torch
import torch.nn as nn
from lib.backbone import get_backbone
from lib.model.classifier import Classifier
from lib.model.regressor import Regressor
from lib.utils import interpolate, scale_coords

class Refiner(nn.Module):
    def __init__(self, opt):
        super(Refiner, self).__init__()

        self.opt = opt
        self.test = True
        self.name = "Refiner"
        self.backbone = get_backbone(opt.backbone)(opt)
        self.max_disp = opt.max_disp

        self.mlp_classification = Classifier(
            filter_channels=[self.backbone.out_ch, 512, 256, 128, self.max_disp]
        )
        self.mlp_regression = Regressor(
            filter_channels=[self.backbone.out_ch + 1, 128, 64, 1],
        )

    def filter(self, batch: torch.Tensor):
        """
        Params:

        batch: tensor with shape Bx3xHxW
        phase: current phase (train, test)
        """

        self.feat_list = self.backbone(batch)
        _, _, height, width = batch.shape
        self.height = height
        self.width = width

        if self.test:
            self.height *= self.opt.upsampling_factor
            self.width *= self.opt.upsampling_factor

    def query(self, points: torch.Tensor, labels: torch.Tensor = None):
        """Query the 2D space for each pixel coordinate, and get the disp value
        for each pixel.

        Params:

        points: points to query
        labels: disp groundtruths
        """
        if labels is not None:
            self.labels = labels
          
        # Coordinated between [-1, 1]
        u = scale_coords(points[:, 0:1, :], self.width)
        v = scale_coords(points[:, 1:2, :], self.height)
        uv = torch.cat([u, v], 1)
        self.uv = uv

        # Interpolate features
        for i, im_feat in enumerate(self.feat_list):
            interp_feat = interpolate(im_feat, uv)
            features = interp_feat if not i else torch.cat([features, interp_feat], 1)

        # estimate integer part of disparity (classification)
        self.probs = self.mlp_classification(features)
        self.disparity = (
            torch.argmax(self.probs, dim=1, keepdim=True).detach().clone().float()
        )
        offset = self.mlp_regression(torch.cat([features, self.disparity], 1))
        self.disparity = self.disparity + offset

    def get_preds(self):
        coords = torch.arange(
            0,
            self.max_disp,
            device=self.opt.device,
            dtype=torch.float32,
            requires_grad=False,
        )[None, :, None]
        preds = torch.sum(self.probs * coords, 1)
        return preds

    def get_disparity(self):
        return self.disparity

    def get_probs(self):
        return self.probs

    def get_confidence(self):
        confidence,_ = torch.max(self.probs, dim=1, keepdim=True)
        return confidence

    def get_error(self):
        classification_loss = 0
        regression_loss = 0

        # classification loss
        b = 2
        gt_i = torch.arange(0, self.max_disp).to(self.labels.device)
        gt_i = torch.reshape(gt_i, (1, self.max_disp, 1))
        gt = torch.exp(-0.5 * torch.abs(self.labels - gt_i) ** 2 / b) / b
        probs = self.get_probs()
        ce = torch.where(
            probs > 0, -(gt * torch.log(probs + 1e-10)), torch.zeros_like(probs)
        )
        ce = torch.mean(ce, dim=2)
        ce = torch.mean(ce, dim=0)
        classification_loss = torch.sum(ce)

        # subpixel offset
        regression_loss = torch.nn.functional.l1_loss(
            self.disparity, self.labels, reduction="none"
        )
        regression_loss = torch.where(
            regression_loss > 1, torch.zeros_like(regression_loss), regression_loss
        )
        regression_loss = torch.mean(regression_loss)

        errors = {
            "classification": classification_loss,
            "regression": regression_loss
        }
        return errors

    def forward(
        self,
        img: torch.Tensor,
        disp: torch.Tensor,
        points: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        # build the input (1,3 or 4 channels)
        x = disp

        if self.opt.num_in_ch == 4 or self.opt.num_in_ch == 2:
            x = torch.cat([img, x], dim=1)
        if self.opt.num_in_ch == 3:
            x = img

        self.input_disp = disp

        # Get features from inputs
        self.filter(x)

        # Point query
        self.query(points, labels)

        # Get the error
        error = self.get_error()
        return error
