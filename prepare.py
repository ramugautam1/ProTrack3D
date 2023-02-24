# import cv2
import gc
import glob as glob
import math as math
import os
import sys
import time
import matplotlib as plt
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io as scio
from PIL import Image
from skimage.transform import resize
from tifffile import imsave
from functions import niftiread, niftiwriteF

def prepare(imageName, protein1Name, protein2Name):
    # takes the niftii, saves the tif's, saves the 3D images of each channel at all time points
    originalImageName = os.path.basename(imageName)
    originalImageAddress = os.path.dirname(imageName)+'/'
    time = 1;
    im=niftiread(imageName)
    z = np.array(np.shape(im))[2]
    # originalImageName = 'EcadMyo_08'
    # originalImageAddress = '/home/nirvan/Desktop/Projects/EcadMyo_08_all/'
    # originalImage = nib.load(originalImageAddress + originalImageName + '.nii')
    originalImage = niftiread(originalImageAddress+originalImageName)
    originalImage = originalImage + 32768

    # originalImageFloat32 = np.asarray(originalImage.dataobj).astype(np.float32).squeeze()
    originalImageFloat32 = np.asarray(nib.load(originalImageAddress + originalImageName).dataobj).astype(np.float32).squeeze()

    print(np.shape(originalImageFloat32))

    originalImageSize = np.shape(originalImage);
    protein1name = protein1Name
    protein2name = protein2Name

    # if not os.path.isdir(originalImageAddress + "3DImage"):
    #     os.makedirs(originalImageAddress + "3DImage")

    dirp1 = originalImageAddress + '3DImage/' + originalImageName.split('.')[0] + '/' + protein1name
    dirp2 = originalImageAddress + '3DImage/' + originalImageName.split('.')[0] + '/' + protein2name

    if not os.path.isdir(dirp1):
        os.makedirs(dirp1)
    if not os.path.isdir(dirp2):
        os.makedirs(dirp2)

    print(originalImageSize)
    ttag = ''
    for i in range(0, originalImageSize[3]):
        if (i < 9):
            ttag = '00'
        elif (i < 99):
            ttag = '0'
        if np.size(np.shape(originalImage))>4:
            slice1 = originalImage[:, :, :, i, 0]
            slice2 = originalImage[:, :, :, i, 1]

            niftiwriteF(slice1, dirp1 + '/threeDimg_' + ttag + str(i + 1))
            niftiwriteF(slice2, dirp2 + '/threeDimg_' + ttag + str(i + 1))

            for j in range(0, originalImageSize[2]):
                if (j < 9):
                    ztag = '000'
                elif (j < 99):
                    ztag = '00'
                else:
                    ztag = '0'

                tifname1 = dirp1 + '/' + originalImageName + '_t' + ttag + str(i + 1) + '_z' + ztag + str(j + 1) + '.tif'
                tifname2 = dirp2 + '/' + originalImageName + '_t' + ttag + str(i + 1) + '_z' + ztag + str(j + 1) + '.tif'

                Xxxx = Image.fromarray(originalImageFloat32[:, :, j, i, 0], mode='F')
                sliceA = Image.fromarray(np.asarray(Xxxx).squeeze())
                sliceA.save(tifname1)
                Xxxx = Image.fromarray(originalImageFloat32[:, :, j, i, 0], mode='F')
                sliceB = Image.fromarray(np.asarray(Xxxx).squeeze())
                sliceB.save(tifname2)
        else:
            slice1 = originalImage[:, :, :, i]

            niftiwriteF(slice1, dirp1 + '/threeDimg_' + ttag + str(i + 1))

            for j in range(0, originalImageSize[2]):
                if (j < 9):
                    ztag = '000'
                elif (j < 99):
                    ztag = '00'
                else:
                    ztag = '0'

                tifname1 = dirp1 + '/' + originalImageName + '_t' + ttag + str(i + 1) + '_z' + ztag + str(
                    j + 1) + '.tif'

                Xxxx = Image.fromarray(originalImageFloat32[:, :, j, i], mode='F')
                sliceA = Image.fromarray(np.asarray(Xxxx).squeeze())
                sliceA.save(tifname1)


    # del sliceB, slice2, slice1, sliceA, originalImage, originalImageFloat32  # clear; free up some memory
    # gc.collect()
    ####################################################################################################################