from utils import config
from utils.dataset import KITTI
from utils.model import UNet
import torch
import matplotlib.pyplot as plt

import cv2
import numpy as np
import os
from urllib import request, error
import zipfile
import shutil
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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


def prepare_plot(input, label, pred_label):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 10))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(input[:3])
    ax[1].imshow(input[3:])
    ax[2].imshow(label)
    ax[3].imshow(pred_label)
    # set the titles of the subplots
    ax[0].set_title("Image1")
    ax[1].set_title("Image2")
    ax[2].set_title("Original interp")
    ax[3].set_title("Predicted interp")
    # set the layout of the figure and display it
    figure.tight_layout()
    figure.show()


def make_predictions(model, input, label):

    prediction = model(input)

    prepare_plot(input, label, prediction)


def main():

    # make val dataset
    new_data_dir = 'dataset/kitti_raw_validation'
    if not os.path.exists(new_data_dir):
        # make directory where data will be (inside dataset)
        os.mkdir(new_data_dir)

    # loop through each filename to download and unzip data
    file_names = open('dataset/filenames_val.txt', 'r')
    for file in file_names.readlines():
        #download_data(file[:-1])
        #unzip_data(file[:-1])
        print(file)

    #cleaning_data_dirs('dataset/kitti_raw_validation')

    # loading val dataset
    val_sequence_paths = glob.glob(f'{config.VAL_PATH}/*_*')

    # transforms that need to be done on the data when retrieved from the disk as PIL Image
    transform_all = transforms.Compose([
        transforms.Resize((128, 384)),
        transforms.ToTensor(),
    ])

    dataset_val = KITTI(sequence_paths=val_sequence_paths, transform=transform_all)

    val_loader = DataLoader(dataset_val, shuffle=True,
                              batch_size=config.BATCH_SIZE,
                              pin_memory=config.PIN_MEMORY,
                              num_workers=os.cpu_count())

    # load the image paths in our testing file and randomly select 10 image paths
    print("[INFO] loading up test image paths...")
    #imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
    #imagePaths = np.random.choice(imagePaths, size=10)
    # load our model from disk and flash it to the current device
    print("[INFO] load up model...")
    #unet = torch.load(config.MODEL_PATH).to(config.DEVICE)
    #unet = torch.load('/Users/aure/Documents/CARES/feature_matching/fake_out/best_model.pth.tar').to(config.DEVICE)

    # initialise model
    unet = UNet(n_channels=6, n_classes=3, bilinear=False)
    # weights
    checkpoint = torch.load(config.MODEL_PATH)
    # load weights to model
    # unet.load_state_dict(checkpoint['state_dict']).to(config.DEVICE)
    unet.load_state_dict(checkpoint, map_location=torch.device(config.DEVICE))

    # set model in evaluation mode
    unet.eval()

    (input, label) = dataset_val[30]

    input, label = (input.to(config.DEVICE), label.to(config.DEVICE))

    # iterate over the randomly selected test image paths
    for (i_batch, (input_imgs, label)) in enumerate(val_loader):

        # send the input to the device we are training our model on
        (input_imgs, label) = (input_imgs.to(config.DEVICE), label.to(config.DEVICE))

        make_predictions(unet, input_imgs, label)

    return


if __name__ == '__main__':
    main()
