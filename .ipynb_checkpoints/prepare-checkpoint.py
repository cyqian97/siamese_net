import os
import csv
import torch
import random
import itertools
from PIL import Image
from copy import deepcopy
import config_gcp as config
import torchvision.transforms as transforms


# +
def generate_csv(directory,total_number=0):
    rows = []
    sigT = []
    sigF = []
    dir_list  = os.listdir(directory)
    dir_list.sort()
    for directory in dir_list[0:-1:2]:
        for root, dirs, files in os.walk(os.path.join(config.training_dir, directory)):
            sigT = deepcopy(files)
        for root, dirs, files in os.walk(os.path.join(config.training_dir, directory + "_forg")):
            sigF = deepcopy(files)
        for pair in itertools.combinations(sigT, 2):
            rows.append([os.path.join(directory, pair[0]), os.path.join(directory, pair[1]), '0'])
        for pair in itertools.product(sigT, sigF):
            rows.append([os.path.join(directory, pair[0]), os.path.join(directory + "_forg", pair[1]), '1'])
    if 0 < total_number < len(rows):
        rows = random.sample(rows, total_number)
        
    with open('train_data.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerows(rows)
        
def dataset_std(directory,tansform):
    e_x = 0;
    e_xsq = 0;
    pixel_count = 0
    for root, dirs, files in os.walk(config.training_dir):
        for file in files:
            img = Image.open(os.path.join(root,file))
            img = img.convert("L")
            img = 1-transform(img)
            e_x += torch.sum(img)
            e_xsq += torch.sum(torch.pow(img,2))
            pixel_count += img.shape.numel()
    print("mean:",(e_x/pixel_count).numpy())
    print("std:",torch.sqrt(e_xsq/pixel_count-(e_x/pixel_count)**2).numpy())
        
if __name__ == '__main__':
    import config_gcp as config
#     generate_csv(config.training_dir)
    transform=transforms.Compose(
            [transforms.Resize((config.img_height, config.img_width)), transforms.ToTensor()]
        )
    dataset_std(config.training_dir,transform)
