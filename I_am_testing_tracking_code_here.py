import statistics
import scipy.io as scio
import numpy as np
import pandas as pd
from functions import line
from datetime import datetime
import xlsxwriter
import os
import math
import glob as glob
import nibabel as nib
from skimage import measure
import re
from correlation20220708 import correlation
from functions import dashline, starline, niftiread, niftiwrite, niftiwriteF, intersect, setdiff, isempty, rand, nan_2d,niftireadI, niftiwriteu16


def trackStep2(track_op_folder, imageName, protein1Name, protein2Name, initialpoint=1, startpoint=1, endpoint=40,trackbackT=2):
    protein1Name = protein1Name
    protein2Name = protein2Name
    imageFolder = os.path.dirname(imageName)
    imageNameOnly = os.path.basename(imageName)
    starline()
    print('Step 2 Start')
    im = niftiread(imageName)
    sz = np.shape(im)
    I3dw = [sz[0], sz[1], sz[2]]
    padding = [20, 20, 2]
    timm = datetime.now()

    folder = track_op_folder

    if not os.path.isdir(folder):
        print(os.makedirs(folder))

    excelFilename = folder + 'TrackingID' + re.sub(r'\W+', '_', str(timm)) + '.xlsx'

    workbook = xlsxwriter.Workbook(excelFilename)

    worksheet1 = workbook.add_worksheet()
    worksheet2 = workbook.add_worksheet()
    worksheet3 = workbook.add_worksheet()
    worksheet4 = workbook.add_worksheet()
    worksheet5 = workbook.add_worksheet()
    worksheet6 = workbook.add_worksheet()
    worksheet7 = workbook.add_worksheet()
    worksheet8 = workbook.add_worksheet()
    worksheet9 = workbook.add_worksheet()
    worksheet10 = workbook.add_worksheet()
    worksheet11 = workbook.add_worksheet()
    worksheet12 = workbook.add_worksheet()

    worksheet2.write('A1', 'time')  # write titles to excel
    worksheet2.write('B1', 'old')
    worksheet2.write('C1', 'new')
    worksheet2.write('D1', 'split')
    worksheet2.write('E1', 'fusion')

    depth = 64

    xlswriter1 = pd.DataFrame(nan_2d(20000, endpoint * 2))
    xlswriter2 = pd.DataFrame(np.zeros((endpoint*2, 5)))
    xlswriter3 = pd.DataFrame(nan_2d(20000, endpoint * 2))
    xlswriter4 = pd.DataFrame(nan_2d(20000, endpoint * 2))
    xlswriter5 = pd.DataFrame(nan_2d(20000, endpoint * 2))
    xlswriter6 = pd.DataFrame(nan_2d(20000, endpoint * 2))
    xlswriter7 = pd.DataFrame(nan_2d(20000, endpoint * 2))
    xlswriter8 = pd.DataFrame(nan_2d(20000, endpoint * 2))
    xlswriter9 = pd.DataFrame(nan_2d(20000, endpoint * 2))
    xlswriter10 = pd.DataFrame(nan_2d(20000, endpoint * 2))
    xlswriter11 = pd.DataFrame(nan_2d(20000, endpoint * 2))
    xlswriter12 = pd.DataFrame(nan_2d(20000, endpoint * 2))

    print(f'depth={depth}, startpoint={startpoint}, endpoint={endpoint}')

    # the weight decay of 'extended search' (not used right now in correlation calculation)
    spatial_extend_matrix = np.full((10,10,3,depth),0)
    for i1 in range(0, 10):
        for i2 in range(0, 10):
            for i3 in range(0, 3):
                spatial_extend_matrix[i1, i2, i3, :] = math.exp(((i1 + 1 - 5) + (i2 + 1 - 5) + (i3 + 1 - 2)) / 20)

    print(folder)

    for time in range(startpoint, endpoint+1):
        dashline()
        tic = datetime.now()
        t1 = str(time)
        t2 = str(time+1)

        print((f'time point: {t1} --> {t2}'))

        worksheet1.write(0, time * 2 - 2, str(t1))
        worksheet1.write(0, time * 2 - 1, str(t2))
        worksheet3.write(0, time * 2 - 1, str(t2))
        worksheet4.write(0, time * 2 - 1, str(t2))
        worksheet5.write(0, time * 2 - 1, str(t2))
        worksheet6.write(0, time * 2 - 1, str(t2))
        worksheet7.write(0, time * 2 - 1, str(t2))
        worksheet8.write(0, time * 2 - 1, str(t2))
        worksheet9.write(0, time * 2 - 1, str(t2))
        worksheet10.write(0, time * 2 - 1, str(t2))
        worksheet11.write(0, time * 2 - 1, str(t2))
        worksheet12.write(0, time * 2 - 1, str(t2))

        addr1 = folder + t1 + '/'
        addr2 = folder + t2 + '/'

        # calculate correlation between this and next time point, using (labeled images and weights from step 1)
        if time == initialpoint:
            for i1 in range(1, time - initialpoint + 1 + 1):
                # print(addr2)
                Fullsize_2 = niftiread(addr2 + 'Fullsize_label_' + t2 + '.nii')
                Fullsize_regression_2 = niftiread(addr2 + 'Weights_' + t2 + '.nii')
                if i1 == time - initialpoint + 1:
                    # print(addr1)
                    Fullsize_1 = niftiread(addr1 + 'Fullsize_label_' + t1 + '.nii')
                    Fullsize_regression_1 = niftiread(addr1 + 'Weights_' + t1 + '.nii')
                else:
                    Fullsize_1 = niftiread(addr1 + 'Fullsize_2_aftertracking_' + t1 + '.nii')
                    Fullsize_regression_1 = niftiread(addr1 + 'Weights_' + t1 + '.nii')

                correlation(Fullsize_1, Fullsize_2, Fullsize_regression_1, Fullsize_regression_2, t2, i1,
                            spatial_extend_matrix, addr2, padding)
        else:
            for i1 in range(1, trackbackT + 1):
                Fullsize_2 = niftiread(addr2 + 'Fullsize_label_' + t2 + '.nii')
                Fullsize_regression_2 = niftiread(addr2 + 'Weights_' + t2 + '.nii')
                Fullsize_1 = niftiread(addr1 + 'Fullsize_2_aftertracking_' + t1 + '.nii')
                Fullsize_regression_1 = niftiread(addr1 + 'Weights_' + t1 + '.nii')
                correlation(Fullsize_1, Fullsize_2, Fullsize_regression_1, Fullsize_regression_2, t2, i1,
                            spatial_extend_matrix, addr2, padding)

        t1 = str(time)
        t2 = str(time + 1)

        # read the correlation calculation results
        correlation_map_padding_show1 = niftiread(
            folder + t2 + '/' + 'correlation_map_padding_show_traceback1_' + t2 + '.nii')
        correlation_map_padding_hide1 = niftiread(
            folder + t2 + '/' + 'correlation_map_padding_hide_traceback1_' + t2 + '.nii')


        # Reading centroids

        if time - initialpoint < trackbackT and time > initialpoint:
            for i1 in range(1, time - initialpoint + 1):
                Registration1 = niftiread(folder + t1 + '/' + 'Registration2_tracking_' + t1 + '.nii')
                correlation_map_padding_show1_2 = niftiread(
                    folder + t2 + '/' + 'correlation_map_padding_show_traceback' + str(i1) + '_' + t2 + '.nii')
                correlation_map_padding_hide1_2 = niftiread(
                    folder + t2 + '/' + 'correlation_map_padding_hide_traceback' + str(i1) + '_' + t2 + '.nii')

                for i2 in range(0, I3dw[0] + padding[0] * 2):  # 0 because python
                    for i3 in range(0, I3dw[1] + padding[1] * 2):
                        for i4 in range(0, I3dw[2] + padding[2] * 2):
                            if correlation_map_padding_hide1[i2, i3, i4] < correlation_map_padding_hide1_2[i2, i3, i4] \
                                    and correlation_map_padding_show1_2[i2, i3, i4] != 0:
                                correlation_map_padding_show1[i2, i3, i4] = correlation_map_padding_show1_2[i2, i3, i4]

        elif time - initialpoint >= trackbackT and time > initialpoint:
            for i1 in range(2, trackbackT + 1):
                Registration1 = niftiread(folder + t1 + '/' + 'Registration2_tracking_' + t1 + '.nii')
                correlation_map_padding_show1_2 = niftiread(
                    folder + t2 + '/' + 'correlation_map_padding_show_traceback' + str(i1) + '_' + t2 + '.nii')
                correlation_map_padding_hide1_2 = niftiread(
                    folder + t2 + '/' + 'correlation_map_padding_hide_traceback' + str(i1) + '_' + t2 + '.nii')
                for i2 in range(0, I3dw[0] + padding[0] * 2):  # 0 because python
                    for i3 in range(0, I3dw[1] + padding[1] * 2):
                        for i4 in range(0, I3dw[2] + padding[2] * 2):
                            if correlation_map_padding_hide1[i2, i3, i4] < correlation_map_padding_hide1_2[i2, i3, i4] \
                                    and correlation_map_padding_show1_2[i2, i3, i4] != 0:
                                correlation_map_padding_show1[i2, i3, i4] = correlation_map_padding_show1_2[i2, i3, i4]
        else:
            Registration1 = niftiread(folder + t1 + '/' + 'Registration_' + t1 + '.nii')

        # -----------------------------------------------------------Good Until Here---------------------------------------------------------------------------------------------
        # Read segmentation
        Fullsize_2 = niftiread(folder + t2 + '/Fullsize_label_' + t2 + '.nii').astype(int)
        Fullsize_2_2 = np.zeros(shape=(np.shape(Fullsize_2)))

        # crop the expanded sample to its original size
        correlation_map_padding_show2 = correlation_map_padding_show1[padding[0]:-1 * padding[0],
                                        padding[1]:-1 * padding[1],
                                        padding[2]:-1 * padding[2]]
        Fullsize_2_mark = correlation_map_padding_show2

        if time > initialpoint:
            correlation_map_padding_show2_2 = correlation_map_padding_show1_2[20:-1 * padding[0], 20:-1 * padding[1],
                                              2:-1 * padding[2]]
            Fullsize_1 = correlation_map_padding_show2_2
            Fullsize_1[Fullsize_1 == 0] = np.nan

            # if not initial time point, read the fusion data of last time point (saved in the same folder as this time point, t1)
            detector_fusion_old = scio.loadmat(folder + t1 + '/' + 'fusion_tracking_' + t1 + '.mat')

            for i1 in range(1, np.size(detector_fusion_old['detector3_fusion'], axis=0) + 1, 2):
                detector_fusion_old['detector3_fusion'][i1, :] = 0











