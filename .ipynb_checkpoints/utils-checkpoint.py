import numpy as np
import matplotlib.pyplot as plt
import math
import os
import torch
import pandas as pd
from PIL import Image
import os
import csv
import random
import itertools
from copy import deepcopy

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

def decision_stub(train_data):
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

    # print(
    #     "j_star = %d\ntheta_star = %f\npolorization = %d\nEmpirical Error = %f\n" % (
    #     j_star, theta_star, b_star, F_star / m))
    return F_star / m

def sortKeyGenerator(i):
    def sortKey(v):
        return v[i]

    return sortKey

def generate_csv(dir, total_number=0):
    for directory in os.listdir(dir)[0:-1:2]:
        for root, dirs, files in os.walk(os.path.join(dir, directory)):
            sigT = files
        for root, dirs, files in os.walk(os.path.join(dir, directory + "_forg")):
            sigF = files
            print(sigF)
        for pair in itertools.combinations(sigT, 2):
            rows.append([os.path.join(directory, pair[0]), os.path.join(directory, pair[1]), '1'])
        for pair in itertools.product(sigT, sigF):
            rows.append([os.path.join(directory, pair[0]), os.path.join(directory + "_forg", pair[1]), '0'])
    if 0 < total_number < len(rows):
        rows = random.sample(rows, total_number)
    
    with open('train_data.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerows(rows)
