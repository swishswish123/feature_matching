import torch
import os

# base path of dataset
DATASET_PATH = os.path.join("dataset", "kitti_raw")

# define the path to the images and masks dataset
#TRAIN_PATH = os.path.join(DATASET_PATH, "training/image_2")
#VAL_PATH = os.path.join("dataset", "kitti_raw_validation")
VAL_PATH = os.path.join("dataset", "endo_data")

# define the test split
TEST_SPLIT = 0.15

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# ------------- TRAINING PARAMS

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 6
BATCH_SIZE = 64

# define the input image dimensions-image dimensions to which our images should be resized for our model to process
# them
INPUT_IMAGE_WIDTH = 384
INPUT_IMAGE_HEIGHT = 128

# type of loss- 'mse' , 'l1', 'bce'
LOSS = 'mse'
# -------------------- OUTPUT PARAMS

# define the path to the base output_1 directory
BASE_OUTPUT = "output"

# define the path to the output_1 serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_interpolation.pth.tar")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.jpg"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
