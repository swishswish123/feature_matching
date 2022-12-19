import matplotlib.pyplot as plt
import torchvision
import os
import glob
from PIL import Image
import numpy as np
from utils import config
import matplotlib.cm as cm
import cv2


def plot_imgs(image1, label, image2, pnt, pnt1=[], pnt2=[], arrow_length = 100):

    fig, ax = plt.subplots(1,3, figsize=(4*3,4))
    ax[0].imshow(image1, cmap=cm.gray, vmin=0, vmax=image1.max())

    if len(pnt1) != 0:
        #ax[0].scatter(pnt1[0], pnt1[1], marker="*", color="red", s=10)
        ax[0].arrow(x=pnt1[0]-arrow_length, y=pnt1[1]-arrow_length,dx=arrow_length,dy=arrow_length, color="red")

    ax[1].imshow(label, cmap=cm.gray, vmin=0, vmax=255)
    ax[1].scatter(pnt[0], pnt[1], marker=".", color="green", s=10)
    ax[1].arrow(x=pnt[0] - arrow_length, y=pnt[1] - arrow_length, dx=arrow_length, dy=arrow_length, color="red")

    ax[2].imshow(image2, cmap=cm.gray, vmin=0, vmax=image2.max())

    if len(pnt2) != 0:
        #ax[2].scatter(pnt2[0], pnt2[1], marker="*", color="red", s=10)
        ax[2].arrow(x=pnt2[0] - arrow_length, y=pnt2[1] - arrow_length, dx=arrow_length, dy=arrow_length, color="red")

    #plt.savefig(config.PLOT_PATH, dpi=400)
    plt.show()


def sensitivity_map(im, label, pnt):
    pxl = label[pnt[1], pnt[0]]
    # https://stackoverflow.com/questions/61628380/calculate-distance-from-all-points-in-numpy-array-to-a-single-point-on-the-basis
    dists = np.linalg.norm(np.indices(im[:, :, 0].shape, sparse=True) - np.array([pnt[1], pnt[0]]))
    dists_inv = 1-dists/dists.max()
    dists_inv[dists_inv<0.8] = 0
    #plt.figure()
    #plt.imshow(dists_inv, cmap='gray')
    #plt.show()

    #der = ((pxl-im)/pxl)
    sim = np.dot((pxl-im),1/pxl)
    cos_sim = np.tensordot(pxl, im, axes=[[0], [2]]) / (np.linalg.norm(pxl) * np.linalg.norm(im))
    #plt.figure()
    #plt.imshow(cos_sim, cmap='gray', vmin=0, vmax=1)
    #plt.show()

    return cos_sim * dists_inv


def sensitivity_map_der(im, label, pnt):
    pxl = label[pnt[1], pnt[0]]
    der = ((pxl - im) / pxl * 255).astype(np.uint8)
    return der


def main():

    image1_pth = '/Users/aure/Documents/CARES/feature_matching/dataset/kitti_raw/2011_09_26_drive_0011_sync/image_02/data/0000000067.png'
    image2_pth = '/Users/aure/Documents/CARES/feature_matching/dataset/kitti_raw/2011_09_26_drive_0011_sync/image_02/data/0000000069.png'
    label_pth = '/Users/aure/Documents/CARES/feature_matching/dataset/kitti_raw/2011_09_26_drive_0011_sync/image_02/data/0000000068.png'

    image1 = np.asarray(Image.open(image1_pth))
    image2 = np.asarray(Image.open(image2_pth))
    label = np.asarray(Image.open(label_pth))
    joined_im = np.concatenate([image1, image2], axis=1)
    #pnt = [695, 107]
    #pnt = [600, 100]
    # --- pnts = [[50, 200], [400, 56], [75, 89]]
    p1 = np.random.random_integers(0, high=image1.shape[1]-1, size=(20))
    p2 = np.random.random_integers(0, high=image1.shape[0]-1, size=(20))
    pnts = np.array([p1, p2]).T
    matches = []
    X1 , X2 , Y1, Y2 = [],[],[],[]
    for pnt in pnts :
        #plot_imgs(image1, label, image2, pnt)

        sensitivity_map_1 = sensitivity_map(image1, label, pnt)
        sensitivity_map_2 = sensitivity_map(image2, label, pnt)

        # x, y = np.where(sensitivity_map_1 == np.max(sensitivity_map_1))

        y1,x1 = np.where(sensitivity_map_1 == sensitivity_map_1.max())
        y2,x2 = np.where(sensitivity_map_2 == sensitivity_map_2.max())

        pnt1 = [x1[0],y1[0]]
        pnt2 = [x2[0]+image1.shape[1],y2[0]]
        #plot_imgs(image1, label, image2,pnt,  pnt1, pnt2)

        matches.append([pnt1, pnt2])
        X1.append(x1[0])
        X2.append(x2[0]+image1.shape[1])
        Y1.append(y1[0])
        Y2.append(y2[0])
        #plot_imgs(sensitivity_map_1, label, sensitivity_map_2, pnt)

    print(matches)

    X = np.array([X1, X2])
    Y = np.array([Y1, Y2])
    plt.imshow(joined_im)
    plt.plot(X,
             Y,
             color='red')
    plt.show()


if __name__=='__main__':
    main()