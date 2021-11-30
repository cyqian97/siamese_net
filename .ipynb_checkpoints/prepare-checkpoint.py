import os
import csv
import config
import random
import itertools
from copy import deepcopy


def generate_csv(dir,total_number=0):
    rows = []
    sigT = []
    sigF = []
    dir_list  = os.listdir(config.training_dir)
    dir_list.sort()
    for directory in dir_list[0:-1:2]:
        for root, dirs, files in os.walk(os.path.join(config.training_dir, directory)):
            sigT = deepcopy(files)
        for root, dirs, files in os.walk(os.path.join(config.training_dir, directory + "_forg")):
            sigF = deepcopy(files)
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
if __name__ == '__main__':
    import config_gcp as config
    generate_csv(config.training_dir)


