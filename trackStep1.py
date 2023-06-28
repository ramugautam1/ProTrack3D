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

from functions import niftiwrite, dashline, starline, niftiwriteF,niftiread, niftiwriteu16


def trackStep1(segmentationOutputAddress, trackingOutputAddress, startTime, endTime,imageName):

    starline()
    print('                    step 1 begin          ')
    starline()
    tictic = datetime.now()

    t1 = startTime
    t2 = endTime
    # t2 = 1
    # size of image, size of cuboids
    image = niftiread(imageName)

    I3dw = np.array(np.shape(image))[:3]
    # I3dw = [512, 280, 15]
    # I3d = [32, 35, I3dw[2]]
    I3d = [32,32,I3dw[2]]

    segAddr = segmentationOutputAddress
    trackAddr = trackingOutputAddress

    for time in range(t1, t2 + 1):
        tic = theTime.perf_counter()
        tt = str(time)

        addr = segAddr + tt + '/'
        addr2 = trackAddr + str(time) + '/'

        if not os.path.isdir(addr2):
            os.makedirs(addr2)
        Files1 = sorted(glob.glob(addr + '*.nii'))
        # print(Files1)
        Fullsize = np.zeros(shape=(I3dw[0],I3dw[1],I3dw[2]))
        Fullsize_input = np.zeros(shape=(I3dw[0],I3dw[1],I3dw[2]))
        Weights = np.zeros(shape=(I3dw[0],I3dw[1], I3dw[2], 64))

        c_file = 0

        for i1 in range(0, I3dw[0], I3d[0]):
            for i2 in range(0, I3dw[1], I3d[1]):

                V_arr = np.asarray(nib.load(Files1[c_file]).dataobj).astype(np.float32).squeeze()
           
                V_arr = 1 - V_arr
                V2_arr = np.uint8(V_arr * 255)

                a = i1
                b = i1 + I3d[0]
                c = i2
                d = i2 + I3d[1]


                Fullsize[a:b, c:d, :] = V2_arr

                V_arr = np.asarray(nib.load(Files1[c_file + 1]).dataobj).astype(np.float32).squeeze()

                for iy in range(0, 64):
                    V2_arr = V_arr[:, :, :, iy]
                    # V3_arr = resize(V2_arr, I3d, order=0)
                    Weights[a:b, c:d, :, iy] = V2_arr

                V_arr = np.asarray(nib.load(Files1[c_file + 2]).dataobj)#.astype(np.float32).squeeze()
                # V3_arr = resize(V_arr, I3d, order=0)
                Fullsize_input[a:b, c:d, :] = V2_arr.squeeze()
                c_file = c_file + 4


        #Remove small itty bitty masks
        Fullsize2 = Fullsize.astype(bool)

        Fullsize2 = np.double(morphology.remove_small_objects(Fullsize2, 3))

        stack_after = Fullsize2

        y = np.size(Fullsize, 0)
        x = np.size(Fullsize, 1)
        z = np.size(Fullsize, 2)

        stack_after_BW = stack_after.astype(bool)

        stack_after_label, orgnum = measure.label(stack_after, connectivity=1, return_num=True)
        CC = cc3d.connected_components(stack_after_label, connectivity=6)

        # stats1 = measure.regionprops_table(stack_after_label, properties=('label', 'bbox', 'centroid'))
        stats1 = pd.DataFrame(measure.regionprops_table(CC, properties=('label', 'bbox', 'centroid', 'coords')))

        nib.save(nib.Nifti1Image(np.uint32(stack_after_label), affine=np.eye(4)), addr2 + 'Fullsize_label_' + tt + '.nii')

        niftiwrite(Fullsize2, addr2 + 'Fullsize' + '_' + tt + '.nii')
        FS2 = Fullsize2.copy()
        FS2 = FS2*65535;
        FS2 = np.uint16(FS2)
        niftiwriteu16(FS2,addr2 + 'FS' + '_' + tt + '.nii')

        # # code to save 3d figure
        # plt.rcParams['figure.figsize'] = (10, 10)
        # plt.rcParams['figure.dpi'] = 500
        # default_cycler = cycler(color=[[1, 0, 0, 0.25], [0, 1, 0, 0.5], [0, 0, 1, 0.5], [0, 1, 1, 0.5]])
        # plt.rc('axes', prop_cycle=default_cycler)
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_box_aspect((350,280,20))


        VoxelList = stats1.coords

        # myCube = np.zeros(shape=(512,280,15))
        # for i in range(0,voxels.shape[0]):
        #     # c1 = [250-i, 100, 100] if i<255 else [110,300-i,110]
        #     s=str(i+1)
        #     for j in range(0,np.size(voxels.VoxelList[i],axis=0)):
        #         myCube[voxels.VoxelList[i][j][0], voxels.VoxelList[i][j][1], voxels.VoxelList[i][j][2]] = i
        #     ax.text(voxels.VoxelList[i][j][0] + 1, voxels.VoxelList[i][j][1] + 1, voxels.VoxelList[i][j][2] + 1, s,
        #             (0, 1, 0), fontsize=5, color = 'red')
        #
        # ax.voxels(myCube)
        Registration = []

        # myCube = np.zeros(shape=(512, 280, 15))

        for i in range(0, VoxelList.shape[0]):
            value = i
            Registration.append([value, stats1['centroid-0'][i], stats1['centroid-1'][i], stats1['centroid-2'][i]])

            # s = str(i+1)
            # for j in range(0, np.size(VoxelList[i], axis=0)):
            #     myCube[VoxelList[i][j][0], VoxelList[i][j][1], VoxelList[i][j][2]] = i
            #
            #
            # # ax.text(VoxelList[i][j][0] + 1, VoxelList[i][j][1] + 1, VoxelList[i][j][2] + 1, s,
            # #    .text(505,280,14, str(VoxelList.shape[0]), (1     (0, 1, 0), fontsize=5, color='red')
            # # ax,1,1), fontsize=10, color='blue')


        #
        # for ww in range(512):
        #     for xx in range(280):
        #         for zz in range(15):
        #             myCube[ww, xx, zz] = originalImage[ww,xx,zz,time,1]

        # ax.voxels(myCube)

        # print('\nSaving Files...')
        # plt.show()

        # fig.savefig(addr2 + str(time) + '_3Dconnection2' + '.png')

        niftiwriteF(Weights, addr2 + 'Weights_' + tt + '.nii')

        niftiwriteF(np.array(Registration), addr2 + 'Registration_' + tt + '.nii')
        toc = theTime.perf_counter()
        remaining = round((toc - tic) * (t2 - time))
        print(f'\rProcessing timepoint : {time}/{t2}. total objects: {len(VoxelList)}. Step 1 estimated to complete in {remaining // 60} min {remaining % 60} sec  ',
            end='', flush=True)

        # print(gc.get_count())
        # del myCube, Fullsize, Fullsize_regression, Fullsize2, Fullsize_input, Weights, fig, stack_after, stack_after_BW, stack_after_label, tic, toc, Registration, VoxelList, orgnum, CC, addr, addr2
        # gc.collect()
        # print(gc.get_count())


    toctoc = datetime.now()
    print(f'Step 1 completed in {toctoc-tictic}')
