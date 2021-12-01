import math
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from os.path import join
import config_gcp as config
from contrastive import contrastiveLoss_func
from utils import show_plot,decision_stub
from torch.nn.functional import pairwise_distance


# create a siamese network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
            # modifications
            nn.Flatten(),
            nn.Linear(108800, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True)

        )

        # # Defining the fully connected layers
        # self.fc1 = nn.Sequential(
        #     nn.Linear(30976, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(p=0.5),
        #     nn.Linear(1024, 128),
        #     nn.ReLU(inplace=True),
        # )

    def forward_once(self, x):
        # Forward pass
        # output = self.cnn1(x)
        # output = output.view(output.size()[0], -1)
        # output = self.fc1(output)
        output = self.cnn1(x)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2


# +
# train the model
def training(train_dataloader, valid_dataloader, optimizer, net, criterion):
    loss_min = math.inf
    valid_er_min = math.inf
#     iteration_number = 0
    for epoch in range(1, config.epochs):
        print("Epoch ",epoch," training")
        for i, data in enumerate(tqdm(train_dataloader), 0):
            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()

        
        # Empirical error on the validation set
        valid_distance = np.zeros((valid_dataloader.__len__(),2))
        print("Epoch ",epoch," validating")
        valid_loss = 0
        for i, data in enumerate(tqdm(valid_dataloader),0):
            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            output1, output2 = net(img0, img1)
            valid_distance[i,0] = pairwise_distance(output1,output2).detach().cpu().numpy()
            valid_distance[i,1] = 1-label.detach().cpu().numpy()*2
            valid_loss += contrastiveLoss_func(output1, output2, label,
                                              margin = criterion.margin,
                                               alpha = criterion.alpha,
                                               beta = criterion.beta
                                              ).detach().cpu().numpy()
            
        print("Epoch {}\t Train loss {}".format(epoch, loss_contrastive.item()))    
        
        # Use decision stub to find the best threshhold
        valid_er = decision_stub(valid_distance.tolist(),verbose=True)
        
#         valid_loss = 0
#         for loss in valid_distance.tolist():
#             if loss[-1]==1: valid_loss+=loss[0]**2/2
#             if loss[-1]==-1: valid_loss+=min(0,1-loss[0])**2/2

        print("Validation loss: %.4f"%(valid_loss/valid_dataloader.__len__()))


        
        # Save state_dict if there is any improvement
        if epoch>1:
            if valid_er < valid_er_min:
                valid_er_min = valid_er
                d = optimizer.state_dict()['param_groups'][0]
                torch.save(net.state_dict(),
                           join("state_dict",
                                "AdamW"+ " lr-"+str(d['lr'])
                                + " wd-"+str(d['weight_decay'])
                                + " batch_size-" + str(train_dataloader.batch_size)
                                + " loss-" + str(loss_contrastive.item())
                                + "validation_error" + str(valid_er)
                                + ".pth"))
                print("new model saved")
            elif loss_contrastive.item() < loss_min:
                loss_min = loss_contrastive.item()
                d = optimizer.state_dict()['param_groups'][0]
                torch.save(net.state_dict(),
                           join("state_dict",
                                "AdamW"+ " lr-"+str(d['lr'])
                                + " wd-"+str(d['weight_decay'])
                                + " batch_size-" + str(train_dataloader.batch_size)
                                + " loss-" + str(loss_contrastive.item())
                                + " validation_error" + str(valid_er)
                                + ".pth"))
                print("new model saved")
        else:
            loss_min = loss_contrastive.item()
            valid_er_min = valid_er
    return net
