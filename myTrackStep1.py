import gc
import glob as glob
import cv2
import math as math
import os
import sys
import time as theTime
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io as scio
from PIL import Image
from skimage.transform import resize
from skimage import measure
from skimage import morphology
import matplotlib as mpl
import cc3d
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from cycler import cycler
import skimage
from functions import niftiwrite, dashline, starline, niftiwriteF, niftiread, niftiwriteu16, niftireadu16, niftireadu32


def myTrackStep1(segmentationOutputAddress, trackingOutputAddress, startTime, endTime, imageNameS, imageNameO):
    starline()
    print('                    step 1 begin          ')
    starline()
    tictic = datetime.now()
    imageName=imageNameS
    t1 = startTime
    t2 = endTime
    image = niftiread(imageName)
    I3dw = np.array(np.shape(image))[:3]
    I3d = [32, 32, I3dw[2]]

    segAddr = segmentationOutputAddress
    trackAddr = trackingOutputAddress

    segmented = niftireadu32(segAddr + 'CombinedSO/CombinedSO.nii')
    print(segAddr)
    print(trackAddr)
    for time in range(t1, t2 + 1):
        tic = theTime.perf_counter()
        tt = str(time)
        addr = segAddr + tt + '/'
        addr2 = trackAddr + tt + '/'

        if not os.path.isdir(addr2):
            os.makedirs(addr2)

        Files1 = sorted(glob.glob(addr + '*.nii'))
        Fullsize = segmented[:, :, :, time - 1]
        Weights = np.zeros(shape=(I3dw[0], I3dw[1], I3dw[2], 64))

        c_file = 0

        for i1 in range(0, I3dw[0], I3d[0]): # for i in range 0, 512, 320
            for i2 in range(0, I3dw[1], I3d[1]):
                a = i1
                b = i1 + I3d[0]
                c = i2
                d = i2 + I3d[1]

                V_arr = np.asarray(nib.load(Files1[c_file + 1]).dataobj).astype(np.float32).squeeze()

                for iy in range(0, 64):
                    V2_arr = V_arr[:, :, :, iy]
                    # V3_arr = resize(V2_arr, I3d, order=0)
                    Weights[a:b, c:d, :, iy] = V2_arr

                c_file = c_file + 4

        # Normalize Weights
        for chan in range(Weights.shape[3]):
            channel = Weights[:, :, :, chan]
            ch_min = channel.min()
            ch_max = channel.max()
            if ch_max > ch_min:  # avoid division by zero
                Weights[:, :, :, chan] = (channel - ch_min) / (ch_max - ch_min)

        stack_after = Fullsize.copy()

        stack_after_BW = stack_after.astype(bool)

        # stack_after_label, orgnum = measure.label(stack_after, connectivity=1, return_num=True)
        CC = skimage.measure.label(stack_after_BW,connectivity=2)

        # CC = cc3d.connected_components(stack_after_label, connectivity=18)
        #
        # stats1 = pd.DataFrame(measure.regionprops_table(CC, properties=('label', 'bbox', 'centroid', 'coords')))
        stats1 = pd.DataFrame(measure.regionprops_table(CC, properties=('label', 'bbox', 'centroid', 'coords')))

        nib.save(nib.Nifti1Image(np.uint32(CC), affine=np.eye(4)),
                 addr2 + 'Fullsize_label_' + tt + '.nii')

        niftiwrite(Fullsize, addr2 + 'Fullsize' + '_' + tt + '.nii')

        VoxelList = stats1.coords

        Registration = []

        for i in range(0, VoxelList.shape[0]):
            value = stats1.label[i]
            Registration.append([value, stats1['centroid-0'][i], stats1['centroid-1'][i], stats1['centroid-2'][i]])

        niftiwriteF(Weights, addr2 + 'Weights_' + tt + '.nii')

        niftiwriteF(np.array(Registration), addr2 + 'Registration_' + tt + '.nii')

        toc = theTime.perf_counter()

        remaining = round((toc-tic)*(t2-time))

        print(f'\rProcessing timepoint : {time}/{t2}. Total Objects: {len(VoxelList)}. Step 1 estimated to complete in {remaining//60} min {remaining%60} sec            ', end='', flush=True)

    toctoc = datetime.now()

    print(f'\nStep 1 completed in {toctoc - tictic}')
    starline()