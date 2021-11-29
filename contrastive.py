import torch
import numpy as np
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    "Contrastive loss function"

    def __init__(self, margin=1.0, alpha=1e-4, beta=0.75):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            self.alpha * (1 - label) * torch.pow(euclidean_distance, 2)
            + self.beta * (label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive