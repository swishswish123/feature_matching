from utils import config
import glob
import os
from urllib import request, error
import zipfile
import shutil
import time
from tqdm import tqdm
import numpy as np

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
import torch

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


def train(unet, train_loader, loss_function, optimizer):
    # set the model in training mode
    unet.train()

    # initialize the total training loss
    total_train_loss = 0
    # loop over the training set
    # we iterate over our trainLoader dataloader, which provides a batch of samples at a time
    for (i_batch, (input_imgs, label)) in enumerate(train_loader):

        # send the input to the device we are training our model on
        (input_imgs, label) = (input_imgs.to(config.DEVICE), label.to(config.DEVICE))

        # pass input images through unet to get prediction of interpolation
        prediction = unet(input_imgs)

        # compute loss between model prediction and ground truth label
        print(f'label min: {torch.min(label)}, max of label: {torch.max(label)}')
        print(f'label min: {torch.min(prediction)}, max of label: {torch.max(prediction)}')
        loss = loss_function(prediction, label)

        # update the parameters of model
        optimizer.zero_grad() # getting rid of previously accumulated gradients from previous steps
        loss.backward() # backpropagation
        optimizer.step() # update model params

        # add the loss to the total training loss so far
        total_train_loss += loss

    return total_train_loss


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

    dataset_train = KITTI(sequence_paths=training_sequence_paths, transform = transform_all)
    dataset_test = KITTI(sequence_paths=testing_sequence_paths, transform = transform_all)

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

    # ---------- TRAINING
    # calculate steps per epoch for training and test set
    num_steps_train = len(dataset_train) // config.BATCH_SIZE
    num_steps_test = len(dataset_test) // config.BATCH_SIZE

    # initialize a dictionary to store training history
    train_history = {"train_loss": [], "test_loss": []}

    print(f'[INFO] training the network...')

    startTime = time.time()
    # best_epoch , best_accuracy = 0, 0
    for epoch in tqdm(range(config.NUM_EPOCHS)):

        # training unet on data
        total_train_loss = train(unet, train_loader, loss_function, opt)

        # Once we have processed our entire training set,
        # evaluate our model on the test set to monitor test loss and
        # ensure that our model is not overfitting to the training set.


if __name__ == '__main__':
    main()
