# import the necessary libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset,random_split
import matplotlib.pyplot as plt
import torchvision.utils
import config as config
from utils import imshow, show_plot, SiameseDataset
from contrastive import ContrastiveLoss
import torchvision
from model import SiameseNetwork,training


# load the dataset
training_dir = config.training_dir
testing_dir = config.testing_dir
training_csv = config.training_csv
testing_csv = config.testing_csv



# Load the the dataset from raw image folders
siamese_dataset = SiameseDataset(
    training_csv,
    training_dir,
    transform=transforms.Compose(
        [transforms.Resize((config.img_height, config.img_width)), transforms.ToTensor()]
    ),
)

# Split the dataset into train, validation and test sets
num_train = round(0.8*siamese_dataset.__len__())
num_validate = round(0.1*siamese_dataset.__len__())
num_test = siamese_dataset.__len__()-num_train-num_validate
siamese_train, siamese_valid, siamese_test = random_split(siamese_dataset, [num_train,num_validate,num_test])

# Viewing the sample of images and to check whether its loading properly
# vis_dataloader = DataLoader(siamese_dataset, shuffle=True, batch_size=8)
# dataiter = iter(vis_dataloader)
# example_batch = next(dataiter)
# concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
# imshow(torchvision.utils.make_grid(concatenated))
# print(example_batch[2].numpy())




# Load the dataset as pytorch tensors using dataloader
train_dataloader = DataLoader(
    siamese_train, shuffle=True, num_workers=8, batch_size=config.batch_size
)

valid_dataloader = DataLoader(siamese_valid,shuffle=True,num_workers=8,batch_size=1)

test_dataloader = DataLoader(siamese_test, num_workers=8, batch_size=1, shuffle=True)

# Declare Siamese Network
net = SiameseNetwork().cuda()
# Decalre Loss Function
criterion = ContrastiveLoss()
# Declare Optimizer
optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)



def run():
    torch.multiprocessing.freeze_support()
    print('loop')
    # set the device to cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("start training")
    model = training(train_dataloader,valid_dataloader,optimizer,net,criterion)

    # torch.save(model.state_dict(), "model.pt")
    # print("Model Saved Successfully")

    # Load the test dataset
    test_dataset = SiameseDataset(
        training_csv=testing_csv,
        training_dir=testing_dir,
        transform=transforms.Compose(
            [transforms.Resize((config.img_height,config.img_width)), transforms.ToTensor()]
        ),
    )

    count = 0
    for i, data in enumerate(test_dataloader, 0):
        x0, x1, label = data
        # concat = torch.cat((x0, x1), 0)
        output1, output2 = model(x0.to(device), x1.to(device))

        eucledian_distance = F.pairwise_distance(output1, output2)

        if label == torch.FloatTensor([[0]]):
            label = "Original Pair Of Signature"
        else:
            label = "Forged Pair Of Signature"

        # imshow(torchvision.utils.make_grid(concat))
        print("Predicted Eucledian Distance:-", eucledian_distance.item())
        print("Actual Label:-", label)
        count = count + 1
        if count == 100:
            break


if __name__ == '__main__':
    run()
