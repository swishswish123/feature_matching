import PIL.Image
from torch.utils.data import Dataset
import cv2
from torch import concat
import PIL
from utils import config
from torchvision import transforms
import glob
import numpy as np

class KITTI(Dataset):
    def __init__(self, sequence_paths=None, transform=None):
        self.transform = transform
        self.training_triplet_paths = []
        # store the image and mask filepaths, and augmentation
        # transforms
        for sequence_path in sequence_paths:
            # in each sequence there is many folders. We are interested in the image_02
            # and image_03 which are the RGB. They represent stereo images so we simply take
            # each as a new training example.
            for mono_folder in ['image_02', 'image_03']:
                img_paths = sorted(glob.glob(f'{sequence_path}/{mono_folder}/data/*.png'))
                # print(f'found {len(img_paths)} image paths for {sequence_path}/{mono_folder}')
                for idx in range(len(img_paths)-2):
                    image_path_1 = img_paths[idx]
                    label_path = img_paths[idx + 1]  # middle image acts as interpolated version of images
                    image_path_2 = img_paths[idx + 2]
                    self.training_triplet_paths.append([image_path_1, label_path, image_path_2])

    def __len__(self):
        # return the number of total samples contained in the dataset
        # print(f'found {len(self.training_triplet_paths)} examples')
        return int(len(self.training_triplet_paths))

    def __getitem__(self, idx):
        # grab the triplet of training data:
        image_path_1 = self.training_triplet_paths[idx][0]
        label_path = self.training_triplet_paths[idx][1]  # middle image acts as interpolated version of images
        image_path_2 = self.training_triplet_paths[idx][2]

        # load the 3 images from disk, swap its channels from BGR to RGB,
        image1 = PIL.Image.open(image_path_1).convert('RGB')
        image2 = PIL.Image.open(image_path_2).convert('RGB')
        label = PIL.Image.open(label_path).convert('RGB')

        # check to see if we are applying any transformations (eg. resize, convert to tensor etc
        if self.transform:
            image1, image2, label = self.transform(image1), self.transform(image2), self.transform(label)

        # concat first and 3rd images to create input. Output is the middle img (label)
        input = concat([image1, image2])
        return input, label

class ENDO(Dataset):
    def __init__(self, sequence_paths=None, transform=None):
        self.transform = transform
        self.training_triplet_paths = []
        # store the image and mask filepaths, and augmentation
        # transforms
        for sequence_path in sequence_paths:
            #for mono_folder in ['image_02', 'image_03']:
            img_paths = sorted(glob.glob(f'{sequence_path}/*.*'))
            print(f'found {len(img_paths)} image paths for {sequence_path}')
            for idx in range(len(img_paths)-300): # 2
                image_path_1 = img_paths[idx]
                label_path = img_paths[idx + 150] # 1 # middle image acts as interpolated version of images
                image_path_2 = img_paths[idx + 300] # 2
                self.training_triplet_paths.append([image_path_1, label_path, image_path_2])

    def __len__(self):
        # return the number of total samples contained in the dataset
        # print(f'found {len(self.training_triplet_paths)} examples')
        return int(len(self.training_triplet_paths))

    def __getitem__(self, idx):
        # grab the triplet of training data:
        image_path_1 = self.training_triplet_paths[idx][0]
        label_path = self.training_triplet_paths[idx][1]  # middle image acts as interpolated version of images
        image_path_2 = self.training_triplet_paths[idx][2]

        # load the 3 images from disk, swap its channels from BGR to RGB,
        image1 = PIL.Image.open(image_path_1).convert('RGB')
        image2 = PIL.Image.open(image_path_2).convert('RGB')
        label = PIL.Image.open(label_path).convert('RGB')

        # check to see if we are applying any transformations (eg. resize, convert to tensor etc
        if self.transform:
            image1, image2, label = self.transform(image1), self.transform(image2), self.transform(label)

        # concat first and 3rd images to create input. Output is the middle img (label)
        input = concat([image1, image2])
        return input, label




class ENDO_VIDEO(Dataset):
    def __init__(self, video_paths=None, transform=None):
        self.transform = transform
        self.training_triplet_paths = []
        # store the image and mask filepaths, and augmentation
        # transforms

        skip_factor = 1

        for video_path in video_paths:
            vid_name = video_path.split('/')[-1]

            # count num frames in vid
            video = cv2.VideoCapture(video_path)
            num_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f'found {num_of_frames} frames for video {vid_name}')

            
            images_1 = np.arange(0,num_of_frames-2)
            images_2 = np.arange(1,num_of_frames-1)
            images_3 = np.arange(2,num_of_frames)
            vid_path_name = [video_path]*(num_of_frames-2)

            self.training_triplet_paths += list(zip(vid_path_name,images_1, images_2, images_3))

        video.release()  

    def __len__(self):
        # return the number of total samples contained in the dataset
        # print(f'found {len(self.training_triplet_paths)} examples')
        return int(len(self.training_triplet_paths))

    def __getitem__(self, idx):
        
        print(f'index {idx}')
        # grab the triplet of training data:
        vid_path_name = self.training_triplet_paths[idx][0]

        cap = cv2.VideoCapture(vid_path_name)

        image_frame_1 = self.training_triplet_paths[idx][1]
        label_frame = self.training_triplet_paths[idx][2]  # middle image acts as interpolated version of images
        image_frame_2 = self.training_triplet_paths[idx][3]

        # load the 3 images from disk, swap its channels from BGR to RGB,
        image1 = cap.set(cv2.CAP_PROP_POS_FRAMES, image_frame_1) # .convert('RGB')
        res, image1 = cap.read()
        image2 = cap.set(cv2.CAP_PROP_POS_FRAMES, image_frame_2)
        res, image2 = cap.read()
        label = cap.set(cv2.CAP_PROP_POS_FRAMES, label_frame) # .COLOR_BGR2RGB
        res, label = cap.read()

        # convert to PIL
        image1 = PIL.Image.fromarray(image1).convert('RGB')
        image2 = PIL.Image.fromarray(image2).convert('RGB')
        label = PIL.Image.fromarray(label).convert('RGB')

        # check to see if we are applying any transformations (eg. resize, convert to tensor etc
        if self.transform:
            image1, image2, label = self.transform(image1), self.transform(image2), self.transform(label)

        # concat first and 3rd images to create input. Output is the middle img (label)
        input = concat([image1, image2])

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        return input, label