import os
import csv
import config
import random
import itertools


def generate_csv(total_number=0):
    with open('train_data.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        rows = []
        for directory in os.listdir(config.training_dir)[0:-1:2]:
            for root, dirs, files in os.walk(os.path.join(config.training_dir, directory)):
                sigT = files
            for root, dirs, files in os.walk(os.path.join(config.training_dir, directory + "_forg")):
                sigF = files
            for pair in itertools.combinations(sigT, 2):
                rows.append([os.path.join(directory, pair[0]), os.path.join(directory, pair[1]), '1'])
            for pair in itertools.product(sigT, sigF):
                rows.append([os.path.join(directory, pair[0]), os.path.join(directory + "_forg", pair[1]), '0'])
        if 0 < total_number < len(rows):
            rows = random.sample(rows, total_number)
        spamwriter.writerows(rows)
