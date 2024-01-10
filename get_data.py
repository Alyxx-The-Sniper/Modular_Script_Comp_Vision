import shutil
from shutil import copy
import argparse
from tqdm import tqdm
import os
import random
from collections import defaultdict

# argparse instance
parser = argparse.ArgumentParser()

# arg for food_list and percentage to copy
parser.add_argument('--food_list', default='cup_cakes,donuts,french_fries,ice_cream', type=str)
parser.add_argument('--copy_percentage', default=0.1, type=float)

# get our arguments from the parser
args = parser.parse_args()

# Setup hyperparameters
FOOD_LIST = args.food_list.split(',')
COPY_PERCENTAGE = args.copy_percentage

print(f'[INFO] Copying {COPY_PERCENTAGE} percent of {FOOD_LIST}')



############################################
def download_data_V2(txtfile,
                     source,
                     destination,
                     percentage: float,
                     class_list=[],
                     random_seed=42):
    """ Downloads a sample list from a text file.

    Args:
        txtfile: A link to a txtfile containing data sample name list.
        source: The source destination of the images
        destination: The destination directory to save the downloaded images
        percentage: Percentage to download
        class_list: List of food class to be downloaded.
        random_seed: Default is 42
    """

    random.seed(random_seed)
    class_types = defaultdict(list)
    with open(txtfile, "r") as file:
        lines = [line.strip() for line in file.readlines()]
        for l in lines:
            class_type = l.split("/")
            if class_type[0] in class_list:
                class_types[class_type[0]].append(class_type[1] + ".jpg")

    for i in class_types.keys():
        print("  " + i, end="  ")
        if not os.path.exists(os.path.join(destination, i)):
            os.makedirs(os.path.join(destination, i))

        # Retrieve random sample of % of the data
        data_sample = random.sample(class_types[i], k=int(len(class_types[i]) * percentage))

        with tqdm(total=len(data_sample), ncols=80) as pbar:
            for n in data_sample:
                copy(os.path.join(source, i, n), os.path.join(destination, i, n))
                pbar.update(1)

##########################################################
# Function Call
print('Downloading test data...')
download_data_V2('/content/drive/MyDrive/Deep_learning/data2/meta/meta/test.txt',
                 '/content/drive/MyDrive/Deep_learning/data2/images',
                 os.path.abspath('data/my_fav_foods/test'),
                 percentage=COPY_PERCENTAGE,
                 class_list=FOOD_LIST)

# Train data
print('Downloading train data...')
download_data_V2('/content/drive/MyDrive/Deep_learning/data2/meta/meta/train.txt',
                 '/content/drive/MyDrive/Deep_learning/data2/images',
                 os.path.abspath('data/my_fav_foods/train'),
                 percentage=COPY_PERCENTAGE,
                 class_list=FOOD_LIST)
print('Download Complete')
