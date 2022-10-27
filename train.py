from utils import config
import glob
import os
from urllib import request, error
import zipfile
import shutil
import torchvision.transforms as transforms

from utils import config
from utils.dataset import KITTI

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

if __name__ == '__main__':
    main()
