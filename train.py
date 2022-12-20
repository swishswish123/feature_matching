from utils import config
import glob
import os
from urllib import request, error
import zipfile
import shutil
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
import torch

from utils import config
from utils.dataset import KITTI, ENDO, ENDO_VIDEO
from utils.model import UNet
from torchmetrics import StructuralSimilarityIndexMeasure


def seed_everything(seed=42):
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SSIM_Loss_Lib(nn.Module):
    def __init__(self, data_range=1):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range)

    def forward(self, img1, img2):
        return 1 - self.ssim(img1, img2)


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
    # total_train_loss = 0

    ssim = StructuralSimilarityIndexMeasure(data_range=1)

    train_loss = []
    ssim_all = []

    # loop over the training set
    # we iterate over our trainLoader dataloader, which provides a batch of samples at a time
    for (i_batch, (input_imgs, label)) in enumerate(train_loader):
        # send the input to the device we are training our model on
        (input_imgs, label) = (input_imgs.to(config.DEVICE), label.to(config.DEVICE))

        # pass input images through unet to get prediction of interpolation
        prediction = unet(input_imgs)

        # compute loss between model prediction and ground truth label
        # print(f'label min: {torch.min(label)}, max of label: {torch.max(label)}')
        # print(f'label min: {torch.min(prediction)}, max of label: {torch.max(prediction)}')
        loss = loss_function(prediction, label)

        # update the parameters of model
        optimizer.zero_grad()  # getting rid of previously accumulated gradients from previous steps
        loss.backward()  # backpropagation
        optimizer.step()  # update model params

        # add the loss to the total training loss so far
        #total_train_loss += loss.item()
        train_loss.append(loss.item())
        ssim_all.append(ssim(prediction.detach().cpu(), label.detach().cpu()).item())

    return np.mean(train_loss), np.mean(ssim_all)


def test(unet, test_loader, loss_function):
    # set the model in evaluation mode
    unet.eval()
    ssim = StructuralSimilarityIndexMeasure(data_range=1)

    test_loss = []
    ssim_all = []

    # switch off gradient computation (as during testing we don't want to get weights.
    with torch.no_grad():
        # loop over the validation set
        for (input_imgs, label) in test_loader:
            # send the input and label to the device
            (test_input_imgs, label) = (input_imgs.to(config.DEVICE), label.to(config.DEVICE))

            # make the predictions and calculate the validation loss
            prediction = unet(test_input_imgs)
            loss = loss_function(prediction, label)
            test_loss.append(loss.item())
            ssim_all.append(ssim(prediction.detach().cpu(), label.detach().cpu()).item())

    return np.mean(test_loss), np.mean(ssim_all)


def get_kitti_data():
    new_data_dir = 'dataset/kitti_raw'
    if not os.path.exists(new_data_dir):
        print('creating dir...')
        # make directory where data will be (inside dataset)
        os.mkdir('dataset')
        os.mkdir(new_data_dir)

    if len(os.listdir(new_data_dir)) == 0:
        # loop through each filename to download and unzip data
        file_names = open('filenames.txt', 'r')
        for file in file_names.readlines():
            download_data(file[:-1])
            unzip_data(file[:-1])
            print(file)

        # move all files to corresponding folder
        cleaning_data_dirs(new_data_dir)

def main():
    print('seeding...')
    seed_everything()

    print('downloading data...')

    load_weights = True

    # transforms that need to be done on the data when retrieved from the disk as PIL Image
    transform_all = transforms.Compose([
        transforms.Resize((128, 384)),
        transforms.ToTensor(),
    ])

    # loading train and test dataset
    dataset_paths = glob.glob(f'{config.DATASET_PATH}/*_*/*')
    
    # splitting to test and train sequences
    if len(dataset_paths) == 1:
        print('only one vid found')
        training_sequence_paths = dataset_paths
        testing_sequence_paths = dataset_paths
    else:
        split_loc = int(len(dataset_paths) / 2)  # finding location where to split train and test
        training_sequence_paths = dataset_paths[:split_loc]
        testing_sequence_paths = dataset_paths[split_loc:]

    # -------- DATA PREP
    if config.data == 'kitti_raw':
        # --- DATA DOWNLOAD
        get_kitti_data()

        # loading paths
        dataset_train = KITTI(sequence_paths=training_sequence_paths, transform=transform_all)
        dataset_test = KITTI(sequence_paths=testing_sequence_paths, transform=transform_all)

    elif config.data == 'endo_data':
        
        dataset_train = ENDO(sequence_paths=training_sequence_paths, transform=transform_all)
        dataset_test = ENDO(sequence_paths=testing_sequence_paths, transform=transform_all)
        
    elif config.data == 'endo_videos':
        dataset_train = ENDO_VIDEO(video_paths=training_sequence_paths, transform=transform_all)
        dataset_test = ENDO_VIDEO(video_paths=testing_sequence_paths, transform=transform_all)

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
    # print(unet)

    if load_weights:
        # weights
        checkpoint = torch.load(f'{config.BASE_OUTPUT}/unet_interpolation.pth.tar', map_location=torch.device(config.DEVICE))
        # load weights to model
        unet.load_state_dict(checkpoint['state_dict'])

    # choosing which loss to train UNET with (set this in config file)
    if config.LOSS == 'mse':
        criterion = nn.MSELoss()
    elif config.LOSS == 'l1':
        criterion = nn.L1Loss()
    elif config.LOSS == 'bce':
        criterion = nn.BCEWithLogitsLoss()

    # initialize loss function and optimizer
    loss_function = criterion.to(config.DEVICE)
    opt = AdamW(unet.parameters(), lr=config.INIT_LR)

    # ---------- TRAINING
    # calculate steps per epoch for training and test set
    num_steps_train = len(dataset_train) // config.BATCH_SIZE
    num_steps_test = len(dataset_test) // config.BATCH_SIZE

    # initialize a dictionary to store training history
    train_history = {"train_loss": [], "test_loss": [], "train_ssim":[], 'test_ssim':[]}

    print(f'[INFO] training the network...')

    start_time = time.time()
    best_epoch, best_loss = 0, np.inf
    for epoch in tqdm(range(config.NUM_EPOCHS)):

        # training unet on data
        # total_train_loss = train(unet, train_loader, loss_function, opt)
        avg_train_loss, avg_train_ssim = train(unet, train_loader, loss_function, opt)
        # Once we have processed our entire training set,
        # evaluate our model on the test set to monitor test loss and
        # ensure that our model is not overfitting to the training set.
        # total_test_loss = test(unet, test_loader, loss_function)
        avg_test_loss, avg_test_ssim = test(unet, test_loader, loss_function)
        # calculate the average training and validation loss
        #avg_train_loss = total_train_loss / num_steps_train
        #avg_test_loss = total_test_loss / num_steps_test

        # update our training history
        # train_history["train_loss"].append(avg_train_loss.cpu().detach().numpy())
        # train_history["test_loss"].append(avg_test_loss.cpu().detach().numpy())
        train_history["train_loss"].append(avg_train_loss)
        train_history["test_loss"].append(avg_test_loss)

        train_history["train_ssim"].append(avg_train_ssim)
        train_history["test_ssim"].append(avg_test_ssim)

        # training indo
        print(f'[INFO] EPOCH: {epoch + 1}/{config.NUM_EPOCHS}')
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(
            avg_train_loss, avg_test_loss))
        print("Train ssim: {:.6f}, Test ssim: {:.4f}".format(
            avg_train_ssim, avg_test_ssim))

        # saving model
        if avg_test_loss < best_loss:
            print('saving model...')

            checkpoint = {}

            best_loss = avg_test_loss
            checkpoint['best_epoch'] = epoch

            checkpoint['best_train_loss'] = avg_train_loss
            checkpoint['best_train_ssim'] = avg_train_ssim

            checkpoint['best_test_loss'] = avg_test_loss
            checkpoint['best_test_ssim'] = avg_test_ssim

            checkpoint['state_dict'] = unet.state_dict()
            # torch.save(unet.state_dict(), config.MODEL_PATH)
            torch.save(checkpoint, config.MODEL_PATH)

    # display the total time needed to perform the training
    end_time = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        end_time - start_time))

    # plot the training loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(train_history["train_loss"], label="train_loss")
    plt.plot(train_history["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(config.PLOT_PATH)


if __name__ == '__main__':
    main()
