import os
import math
import torch
import csv
import random
import itertools
import numpy as np
import pandas as pd
from PIL import Image
from copy import deepcopy
import config_gcp as config
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class SiameseDataset:
    def __init__(self, training_csv=None, training_dir=None, transform=None):
        # used to prepare the labels and images path
        self.train_df = pd.read_csv(training_csv)
        self.train_df.columns = ["image1", "image2", "label"]
        self.train_dir = training_dir
        self.transform = transform

    def __getitem__(self, index):
        # getting the image path
        image1_path = os.path.join(self.train_dir, self.train_df.iat[index, 0])
        image2_path = os.path.join(self.train_dir, self.train_df.iat[index, 1])

        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        # Apply image transformations
        if self.transform is not None:
            img0 = 1-self.transform(img0)
            img1 = 1-self.transform(img1)

        return (
            img0,
            img1,
            torch.from_numpy(
                np.array([int(self.train_df.iat[index, 2])], dtype=np.float)
            ),
        )

    def __len__(self):
        return len(self.train_df)


# Load the the dataset from raw image folders
def load_dataset(training_dir,training_csv):
    siamese_dataset = SiameseDataset(
    training_csv,
    training_dir,
    transform=transforms.Compose(
        [transforms.Resize((config.img_height, config.img_width)), 
         transforms.ToTensor(),
         transforms.Normalize(mean = 0.060600653, std=0.12516373)]),
    )
    return siamese_dataset


def decision_stub(train_data,verbose=False):
    F_star = math.inf
    m = len(train_data)
    d = len(train_data[0]) - 1
    b_star = 0

    F0_p = sum([1 for data in train_data if data[-1] == 1])
    for j in range(d):
        train_data.sort(key=sortKeyGenerator(j))
        F_p = F0_p
        if F_p < F_star:
            F_star = F_p
            theta_star = train_data[0][j] - 1
            j_star = j
            b_star = 1
        for i in range(m - 1):
            F_p -= train_data[i][d]
            if F_p < F_star and train_data[i][j] != train_data[i + 1][j]:
                F_star = F_p
                theta_star = (train_data[i][j] + train_data[i + 1][j]) / 2
                j_star = j
                b_star = 1
        i = m - 1
        F_p -= train_data[i][-1]
        if F_p < F_star:
            F_star = F_p
            theta_star = train_data[i][j] + 0.5
            j_star = j
            b_star = 1

    F0_n = sum([1 for data in train_data if data[-1] == -1])

    for j in range(d):
        train_data.sort(key=sortKeyGenerator(j))
        F_n = F0_n
        if F_n < F_star:
            F_star = F_n
            theta_star = train_data[0][j] - 1
            j_star = j
            b_star = -1
        for i in range(m - 1):
            F_n += train_data[i][d]
            if F_n < F_star and train_data[i][j] != train_data[i + 1][j]:
                F_star = F_n
                theta_star = (train_data[i][j] + train_data[i + 1][j]) / 2
                j_star = j
                b_star = -1
        i = m - 1
        F_n += train_data[i][-1]
        if F_n < F_star:
            F_star = F_n
            theta_star = train_data[i][j] + 0.5
            j_star = j
            b_star = -1
    if verbose:
            train_data = np.array(train_data)
            print("+1/-1 ratrio:%.2f/%.2f"%(0.5+np.sum(train_data[:,-1])/m/2,0.5-np.sum(train_data[:,-1])/m/2))
            print("+1 features max:%.2f\t min:%.2f\t mean:%.2f\t median %.2f" %(
                np.max(train_data[train_data[:,-1]==1,j_star]),
                np.min(train_data[train_data[:,-1]==1,j_star]),
                np.mean(train_data[train_data[:,-1]==1,j_star]),
                np.median(train_data[train_data[:,-1]==1,j_star])
            ))            
            print("-1 features max:%.2f\t min:%.2f\t mean:%.2f\t median %.2f" %(
                np.max(train_data[train_data[:,-1]==-1,j_star]),
                np.min(train_data[train_data[:,-1]==-1,j_star]),
                np.mean(train_data[train_data[:,-1]==-1,j_star]),
                np.median(train_data[train_data[:,-1]==-1,j_star])
            ))
            
            print(
                "j_star = %d\ttheta_star = %f\tpolorization = %d\tEmpirical Error = %f" % (
                    j_star, theta_star, b_star, F_star / m))
    return F_star / m

def sortKeyGenerator(i):
    def sortKey(v):
        return v[i]

    return sortKey


def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(
            75,
            8,
            text,
            style="italic",
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 10},
        )
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()
