from typing import List

import torch
import torch.nn as nn
from lib.model.utils import Sine

class Classifier(nn.Module):
    def __init__(self, filter_channels: List[int], no_softmax=False):
        super(Classifier, self).__init__()

        self.filters = []
        self.filter_channels = filter_channels
        self.no_softmax = no_softmax
        self.nfeat = filter_channels[0]
        self.layer_1 = nn.Sequential(
            nn.Conv1d(self.nfeat, filter_channels[1], 1), Sine()
        )

        self.layer_2 = nn.Sequential(
            nn.Conv1d(self.nfeat + filter_channels[1], filter_channels[2], 1),
            Sine(),
        )

        self.layer_3 = nn.Sequential(
            nn.Conv1d(self.nfeat + filter_channels[2], filter_channels[3], 1),
            Sine(),
        )

        self.layer_4 = nn.Sequential(
            nn.Conv1d(self.nfeat + filter_channels[3], filter_channels[4], 1),
        )

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.layer_1.apply(weights_init)
        self.layer_2.apply(weights_init)
        self.layer_3.apply(weights_init)
        self.layer_4.apply(weights_init)

    def forward(self, feature: torch.Tensor):
        #tmpy = torch.cat([feature, confidence], 1)
        feat = self.layer_1(feature)
        feat = self.layer_2(torch.cat([feat, feature], 1))
        feat = self.layer_3(torch.cat([feat, feature], 1))
        feat = self.layer_4(torch.cat([feat, feature], 1))
        if not self.no_softmax:
            probs = torch.nn.functional.softmax(feat, dim=1)
        else:
            probs = feat
        return probs
