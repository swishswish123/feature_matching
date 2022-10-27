from utils import config
import glob
import os
from urllib import request, error
import zipfile
import shutil

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW

from utils import config
from utils.dataset import KITTI
from utils.model import UNet

def download_data(file):
    print(f'downloading {file}...')
    try:
        URL = f'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{file}/{file}_sync.zip'
        request.urlretrieve(URL, f'dataset/{file}.zip')
        print('download complete')
    except error.HTTPError:
        print(f'could not find file {file}')

    return


def unzip_data(file):

    try:
        print(f'unzipping {file}...')
        with zipfile.ZipFile(f'dataset/{file}.zip', 'r') as zip_ref:
            zip_ref.extractall('dataset')
        os.remove(f'dataset/{file}.zip')
        print('file unzipped')
    except FileNotFoundError:
        print(f'file {file} not found')


def cleaning_data_dirs(new_data_dir):

    # dirs_02 = glob.glob('dataset/*_*_*/*/image_02/data')
    # dirs_03 = glob.glob('dataset/*_*_*/*/image_03/data')
    # dirs = glob.glob('dataset/*_*_*/*/image_0[2-3]/data')

    dirs = glob.glob('dataset/*_*_*/*')
    for source_file in dirs:
        shutil.move(source_file, new_data_dir)

    # remove empty year dirs we've just moved all the files out of
    main_dirs = glob.glob('dataset/*_*_*')
    for dir in main_dirs:
        os.rmdir(f"{dir}")
    return


def main():

    # --- DATA DOWNLOAD
    # dataset = torchvision.datasets.Kitti('dataset', train=False,transform= None, target_transform= None, transforms= None, download=True)
    new_data_dir = 'dataset/kitti_raw'
    if not os.path.exists(new_data_dir):
        # make directory where data will be (inside dataset)
        os.mkdir(new_data_dir)

        # loop through each filename to download and unzip data
        file_names = open('dataset/filenames.txt','r')
        for file in file_names.readlines():
            download_data(file[:-1])
            unzip_data(file[:-1])
            print(file)

        # move all files to corresponding folder
        cleaning_data_dirs(new_data_dir)

    # -------- DATA PREP
    # load the image and mask filepaths in a sorted manner

    #training_img_paths = sorted(glob.glob(f'{config.DATASET_PATH}/*.png'))
    #testing_img_paths = sorted(glob.glob(f'{config.DATASET_PATH}/*.png'))
    dataset_paths = glob.glob(f'{config.DATASET_PATH}/*_*')
    split_loc = int(len(dataset_paths)/2) # finding location where to split train and test
    training_sequence_paths = dataset_paths[:split_loc]
    testing_sequence_paths = dataset_paths[split_loc:]

    # transforms that need to be done on the data when retrieved from the disk as PIL Image
    transform_all = transforms.Compose([
        transforms.Resize((128, 384)),
        transforms.ToTensor(),
    ])

    dataset_train = KITTI(sequence_paths=training_sequence_paths)
    dataset_test = KITTI(sequence_paths=testing_sequence_paths)

    print(f"[INFO] found {len(dataset_train)} examples in the training set...")
    print(f"[INFO] found {len(dataset_test)} examples in the test set...")

    # create the training and test data loaders

    # We make shuffle = True since we want samples from all classes to be uniformly present in a
    # batch which is important for optimal learning and convergence of batch
    # gradient-based optimization approaches.
    train_loader = DataLoader(dataset_train, shuffle=True,
                              batch_size=config.BATCH_SIZE,
                              pin_memory=config.PIN_MEMORY,
                              num_workers=os.cpu_count())
    test_loader = DataLoader(dataset_test, shuffle=False,
                             batch_size=config.BATCH_SIZE,
                             pin_memory=config.PIN_MEMORY,
                             num_workers=os.cpu_count())

    # initialize our UNet model:
    # # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    unet = UNet(n_channels=6, n_classes=3, bilinear=False).to(config.DEVICE)
    print(unet)

    # choosing which loss to train UNET with (set this in config file)
    if config.LOSS == 'mse':
        criterion = nn.MSELoss()
    elif config.LOSS == 'l1':
        criterion = nn.L1Loss()
    elif config.LOSS == 'bce':
        criterion = nn.BCELoss()

    # initialize loss function and optimizer
    loss_function = criterion.to(config.DEVICE)
    opt = AdamW(unet.parameters(), lr=config.INIT_LR)


if __name__ == '__main__':
    main()
