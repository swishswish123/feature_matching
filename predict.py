from utils import config
from utils.dataset import KITTI, ENDO
from utils.model import UNet
import torch
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
import os
from urllib import request, error
import zipfile
import shutil
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math
import matplotlib.cm as cm
import PIL
import utils.config
from torch import nn


def test(unet,test_loader, loss_function):
    # set the model in evaluation mode
    unet.eval()
    total_test_loss = []

    # switch off gradient computation (as during testing we don't want to get weights.
    with torch.no_grad():
        # loop over the validation set
        for (input_imgs, label) in test_loader:
            # send the input and label to the device
            (test_input_imgs, label) = (input_imgs.to(config.DEVICE), label.to(config.DEVICE))

            # make the predictions and calculate the validation loss
            prediction = unet(test_input_imgs)
            loss = loss_function(prediction, label)
            total_test_loss.append(loss.item())

    return np.mean(total_test_loss)



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


def plot_grads_correspondences(im1,im1_pxl,im1_grad ,matching_score_1,  im3,im3_pxl,im3_grad,matching_score_3, prediction, label_pxl):

    im1 = im1.detach().cpu()
    im3 = im3.detach().cpu()

    num_img = 3
    fig, ax = plt.subplots(2, num_img, figsize=(5 * num_img, 5), subplot_kw=dict(xticks=[], yticks=[]))

    label_pxl_x = label_pxl[1]
    label_pxl_y = label_pxl[0]
    # label
    ax[0, 0].imshow(prediction[0].detach().cpu().permute(1, 2, 0))
    ax[0, 0].plot(label_pxl_x, label_pxl_y, 'o', markersize=7, color='r')
    text = '({},{})'.format(label_pxl_x, label_pxl_y)
    ax[0, 0].text(label_pxl_x - 20, label_pxl_y - 20, text, color='r')
    ax[0, 0].set_title('Interpolated')
    # im1
    ax[0, 1].imshow(im1.permute(1, 2, 0))
    ax[0, 1].plot(im1_pxl[2], im1_pxl[1], 'o', markersize=7, color='r')
    text = '({},{})'.format(im1_pxl[2], im1_pxl[1])
    ax[0, 1].text(im1_pxl[2] - 20, im1_pxl[1] - 20, text, color='r')
    ax[0, 1].text(im1_pxl[2] - 20, im1_pxl[1] - 8, f'MS: {matching_score_1}', color='r')
    ax[0, 1].set_title('Correspondance: 1st frame')
    # im3
    ax[0, 2].imshow(im3.permute(1, 2, 0))
    ax[0, 2].plot(im3_pxl[2], im3_pxl[1], 'o', markersize=7, color='r')
    text = '({},{})'.format(im3_pxl[2], im3_pxl[1])
    ax[0, 2].text(im3_pxl[2] - 20, im3_pxl[1] - 20, text, color='r')
    ax[0, 2].text(im3_pxl[2] - 20, im3_pxl[1] - 8, f'MS: {matching_score_3}', color='r')

    ax[0, 2].set_title('Correspondance: 3rd frame');

    # gradients
    ax[1, 1].imshow(im1_grad.permute(1, 2, 0), cmap=cm.gray, vmin=0, vmax=im1_grad.max())
    ax[1, 1].set_title('Gradient: 1st frame')

    ax[1, 2].imshow(im3_grad.permute(1, 2, 0), cmap=cm.gray, vmin=0, vmax=im3_grad.max())
    ax[1, 2].set_title('Gradient: 3rd frame')
    plt.show()

def plot_matches(im1, im3, X, Y):
    #image1 = np.asarray(Image.open(image1_pth))
    #image2 = np.asarray(Image.open(image2_pth))
    #label = np.asarray(PIL.Image.open(label_pth))
    #joined_im = np.concatenate([image1, image2], axis=1)
    #inputs = input.detach().cpu().permute(1, 2, 0)
    im1 = im1.detach().cpu().permute(1, 2, 0)
    im3 = im3.detach().cpu().permute(1, 2, 0)

    #joined_imgs = inputs.reshape((inputs.shape[0], inputs.shape[1] * 2, inputs.shape[2] // 2))
    joined_im = np.concatenate([im1, im3], axis=1)

    plt.figure()
    plt.imshow(joined_im)
    #plt.plot(max_im1[2], max_im1[1],'o' ,color='red', markersize=7)
    #plt.plot(max_im3[2]+im1.shape[1], max_im3[1], 'o', color='red', markersize=7)
    plt.plot(X,
             Y)
    plt.show()

    return



def calculate_matching_score(max_im1, input_grad_im):

    # finding max and min boundaries of box
    min_idx = max_im1 - 10
    max_idx = max_im1 + 10
    # checking it's within boundaries- if not replace with boundary
    min_idx[min_idx < 0] = 0
    for i in range(1, 3):
        max_idx[max_idx[i] > input_grad_im.shape[i]] = input_grad_im.shape[i]
    # select intensiry vals of box and max intensity (center)
    box_1 = input_grad_im[:, min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]]
    max_intensity = input_grad_im[max_im1[0], max_im1[1], max_im1[2]]
    matching_score = box_1.mean() / max_intensity
    return matching_score


def get_kitti_data():
    # make val dataset
    new_data_dir = 'dataset/kitti_raw_validation'
    if not os.path.exists(new_data_dir):
        # make directory where data will be (inside dataset)
        os.mkdir(new_data_dir)

    # loop through each filename to download and unzip data
    file_names = open('filenames_val.txt', 'r')
    for file in file_names.readlines():
        # download_data(file[:-1])
        # unzip_data(file[:-1])
        print(file)

    cleaning_data_dirs('dataset/kitti_raw_validation')

def main():

    data = 'endo'
    grid_size = [5, 5]

    # transforms that need to be done on the data when retrieved from the disk as PIL Image
    transform_all = transforms.Compose([
        transforms.Resize((128, 384)),
        transforms.ToTensor(),
    ])

    if data=='kitty':
        get_kitti_data()
        # loading val dataset
        val_sequence_paths = glob.glob(f'{config.VAL_PATH}/*_*')
        dataset_val = KITTI(sequence_paths=val_sequence_paths, transform=transform_all)

    elif data == 'endo':
        val_sequence_paths = glob.glob(f'{config.VAL_PATH}/*')
        dataset_val = ENDO(sequence_paths=val_sequence_paths, transform=transform_all)

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
    # initialise model
    unet = UNet(n_channels=6, n_classes=3, bilinear=False)
    # weights
    checkpoint = torch.load(config.MODEL_PATH, map_location=torch.device(config.DEVICE))
    # load weights to model
    unet.load_state_dict(checkpoint['state_dict'])
    #unet.load_state_dict(checkpoint)

    # set model in evaluation mode
    unet.eval()

    # test performance
    #criterion = criterion = nn.MSELoss()
    #loss_function = criterion.to(config.DEVICE)
    #loss = test(unet, val_loader, loss_function)
    #print(loss)

    # select image for image triplet for matching
    (input, label) = dataset_val[60]
    input, label = (input.to(config.DEVICE), label.to(config.DEVICE))

    # Usually pytorch doen't consider input as the gradient parameter as backpropagation
    # doesn't need to calculate gradient for the input.
    # In our case, we need the gradient for the input, so we force the model to generate
    # gradient for the input by pointing them as the learning parameters (torch.nn.parameter)
    input = torch.nn.Parameter(input)
    image1 = input[:3]
    '''
    image3 = input[3:]
    
    # expanding dims of input so that it also has batch as first dim, then making prediction of interpolated image
    prediction = unet(input[None])

    # reverse sigmoid to get logits
    logits = torch.log(prediction / (1 - prediction))
    '''
    # raster grid (4 by 4 pxl (and 3 channels))
    raster_grid = torch.ones((1, 3, grid_size[0], grid_size[1]))

    # define pixel in interpolated image to find the correspondences
    px = np.random.random_integers(0, high=int(image1.shape[1]-grid_size[0]/2), size=(5))
    py = np.random.random_integers(0, high=int(image1.shape[2]-grid_size[1]/2), size=(5))

    print('max p1', image1.shape[1]-grid_size[0]/2)
    print('max p2', image1.shape[2]-grid_size[1]/2)
    label_pnts = np.array([px, py]).T

    matching_scores = []
    X1 , X3 , Y1, Y3 = [],[],[],[]

    for label_pxl in label_pnts:
        ########

        (input, label) = dataset_val[60]
        input, label = (input.to(config.DEVICE), label.to(config.DEVICE))

        # ------------?
        # Enkaoua, Aureand   torch.nn.Parameter
        # Usually pytorch doen't consider input as the gradient parameter as backpropagation
        # doesn't need to calculate gradient for the input.
        # In our case, we need the gradient for the input, so we force the model to generate
        # gradient for the input by pointing them as the learning parameters (torch.nn.parameter)
        input = torch.nn.Parameter(input)
        image1 = input[:3]
        image3 = input[3:]
        # expanding dims of input so that it also has batch as first dim, then making prediction of interpolated image
        prediction = unet(input[None])

        # reverse sigmoid to get logits
        logits = torch.log(prediction / (1 - prediction))

        #####
        # inserting ones at raster grid location
        target_to_gradient = torch.zeros(logits.shape).to(config.DEVICE)

        start_i = abs(label_pxl[0] - math.floor(grid_size[0]/2))
        end_i = start_i + grid_size[0]

        start_j = abs(label_pxl[1] - math.floor(grid_size[1]/2))
        end_j = start_j + grid_size[1]

        target_to_gradient[:, :, start_i:end_i, start_j:end_j] = raster_grid
        target_to_gradient = target_to_gradient.to(config.DEVICE)

        # Main command to generate gradient for the corresponding predicted class or pixels
        logits.backward(target_to_gradient, retain_graph=True)
        input_grad = input.grad.data.cpu()

        ############### DELAY?
        # time.sleep(3)

        # splitting gradients on inputs back to the 2 images
        input_grad_im1 = input_grad[:3,]
        input_grad_im3 = input_grad[3:,]

        # converting to grayscale
        rgb2gray = transforms.Grayscale(num_output_channels=1)
        input_grad_im1, input_grad_im3 = (rgb2gray(input_grad_im1), rgb2gray(input_grad_im3))

        # finding pixels of the maximum gradient in the generated input gradient for both frames
        max_im1 = (input_grad_im1 == torch.max(input_grad_im1)).nonzero()[0]
        max_im3 = (input_grad_im3 == torch.max(input_grad_im3)).nonzero()[0]

        print('pixel in interpolated image:', label_pxl[0], label_pxl[1])
        print('correspondence in img1:', max_im1)
        print('correspondence in img3:', max_im3)

        # matching score is defined as the ratio between the maximum gradient intensity and the mean gradient intensity within a 20 × 20 area around P
        matching_score_1 = calculate_matching_score(max_im1, input_grad_im1)
        matching_score_3 = calculate_matching_score(max_im3, input_grad_im3)

        matching_scores.append([matching_score_1, matching_score_3])

        # points in im 1
        X1.append(max_im1[2])
        Y1.append(max_im1[1] )

        # pointa in im3
        X3.append(max_im3[2] + image1.shape[2])
        Y3.append(max_im3[1] )

        plot_grads_correspondences(image1, max_im1, input_grad_im1, matching_score_1, image3, max_im3, input_grad_im3,matching_score_3,  prediction,
                                   label_pxl)

        #pnt1 = [max_im1[0],max_im1[1]]
        #pnt2 = [max_im3[0]+image1.shape[1],max_im3[1]]

        #matches.append([pnt1, pnt2])


    print(matching_scores)

    X = np.array([X1, X3])
    Y = np.array([Y1, Y3])

    plot_matches(input[:3], input[3:], X, Y)

    '''
    
    # iterate over the randomly selected test image paths
    for (i_batch, (input_imgs, label)) in enumerate(val_loader):

        # send the input to the device we are training our model on
        (input_imgs, label) = (input_imgs.to(config.DEVICE), label.to(config.DEVICE))

        make_predictions(unet, input_imgs, label)
    '''
    return


if __name__ == '__main__':
    main()
