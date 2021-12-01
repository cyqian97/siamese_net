import torch
import numpy as np
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    "Contrastive loss function"

    def __init__(self, margin=1.0, alpha=0.5, beta=0.5):
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
#         print("\ntheir loss: %.4f\t"%(loss_contrastive))
#         loss = 0
#         for i,l in enumerate(label,0):
#             d2 = torch.sum(torch.pow(output1[i]-output2[i],2))
#             if l==0:
#                 loss += d2*self.alpha
#             else: 
#                 loss += torch.pow(torch.min(self.margin - torch.sqrt(d2),0).values,2)*self.beta
#         print("my loss: %.4f\n"%(loss/len(label)))
        return loss_contrastive
