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
# from testCorr import correlation
from functions import dashline, starline, niftiread, niftiwrite, niftiwriteF, intersect, setdiff, isempty, rand, nan_2d


def trackStep2(track_op_folder,  imageName, protein1Name, protein2Name, initialpoint=1, startpoint=1, endpoint=40, trackbackT=2):
    protein1Name = protein1Name
    protein2Name = protein2Name
    imageFolder = os.path.dirname(imageName)
    imageNameOnly = os.path.basename(imageName)
    starline()  # print **************************************
    print('step 2 start')
    starline()
    # colormap = scio.loadmat('/home/nirvan/Desktop/Projects/MATLAB CODES/colormap.mat')
    im = niftiread(imageName)
    sz = np.shape(im)
    I3dw = [sz[0],sz[1],sz[2]]
    # I3dw = [512, 280, 15]
    padding = [20, 20, 2]
    timm = datetime.now()

    folder = track_op_folder
    # trackbackT = 2

    if not os.path.isdir(folder):
        print(os.makedirs(folder))

    excelFilename = folder + 'TrackingID' + re.sub(r'\W+', '_', str(timm)) + '.xlsx'  # the excel file name to write the tracking result

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

    depth = 64  # the deep features to take in correlation calculation
    # initialpoint = 1  # the very first time point of all samples
    # startpoint = 1  # the time point to start tracking
    # endpoint = 10  # the time point to stop tracking

    # Tracking each object
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

    print(f'depth = {depth}, startpoint = {startpoint}, endpoint = {endpoint}')

    spatial_extend_matrix = np.full((10, 10, 3, depth),
                                    0)  # the weight decay of 'extended search' (not used right now in correlation calculation)

    for i1 in range(0, 10):
        for i2 in range(0, 10):
            for i3 in range(0, 3):
                spatial_extend_matrix[i1, i2, i3, :] = math.exp(((i1 + 1 - 5) + (i2 + 1 - 5) + (i3 + 1 - 2)) / 20)

    print(folder)

    for time in range(startpoint, endpoint + 1):
        dashline()
        tic = datetime.now()
        t1 = str(time)
        t2 = str(time + 1)

        print(f'time point: {t1} --> {t2}')

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

        # if not os.path.isdir(folder):
        #     os.makedirs(addr1)
        # if not os.path.isdir(folder):
        #     os.makedirs(addr1)

        Files1 = sorted(glob.glob(addr1 + '*.nii'))
        Files2 = sorted(glob.glob(addr2 + '.nii'))

        # calculate correlation between this and next time point, using (labeled images and weights from step 1)

        # if time - initialpoint < trackbackT:  # calculating correlation for start time points (e.g. time=2)
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
                # dashline()

        else:
            for i1 in range(1, trackbackT + 1):
                Fullsize_2 = niftiread(addr2 + 'Fullsize_label_' + t2 + '.nii')
                Fullsize_regression_2 = niftiread(addr2 + 'Weights_' + t2 + '.nii')
                Fullsize_1 = niftiread(addr1 + 'Fullsize_2_aftertracking_' + t1 + '.nii')
                Fullsize_regression_1 = niftiread(addr1 + 'Weights_' + t1 + '.nii')
                correlation(Fullsize_1, Fullsize_2, Fullsize_regression_1, Fullsize_regression_2, t2, i1,
                            spatial_extend_matrix, addr2, padding)
                # dashline()

        if time > 2:
            del Fullsize_1, Fullsize_regression_1, Fullsize_2, Fullsize_regression_2

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
            # print('here')
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
            # print(detector_fusion_old['detector3_fusion'])

        Fullsize_2_mark[Fullsize_2 == 0] = 0

        # ------------------------------- OK Tested Until Here-----------------------------------

        if time > 1:
            del correlation_map_padding_show1, correlation_map_padding_show1_2, correlation_map_padding_hide1, correlation_map_padding_hide1_2

        # Get the object characteristics
        # Fullsize_2_mark_BW = Fullsize_2_mark
        # Fullsize_2_mark_BW[Fullsize_2_mark_BW > 0] = 1
        # Fullsize_2_mark_BW = Fullsize_2_mark_BW.astype(bool)
        # Fullsize_2_mark_label, orgnum = measure.label(Fullsize_2_mark, connectivity=1, return_num=True)

        # stats1 = regionprops3(Fullsize_2,'BoundingBox','VoxelList','ConvexHull','Centroid');
        stats1 = pd.DataFrame(
            measure.regionprops_table(Fullsize_2.astype(int), properties=('label', 'bbox', 'coords', 'centroid')))
        VoxelList = stats1.coords

        #  sort the objects in descending order of size

        count = np.zeros(stats1.shape[0])
        for i in range(stats1.shape[0]):
            count[i] = np.size(stats1.coords[i], axis=0)
        stats1['Count'] = count.astype(int)

        stats2 = stats1.sort_values(by='Count', axis=0, ascending=False, ignore_index=False)
        print(f'objects found: {np.amax(stats2.label)} previous total: { np.size(Registration1,axis=0)}')

        # -----
        detector_fusion = {}
        detector_split = {}
        detector2_fusion = {}
        detector3_fusion = {}

        # stack_after_label[Fullsize_2_mark > 0] = 0  Not used, not initialized
        Fullsize_2_mark = Fullsize_2_mark.astype(float)

        Fullsize_2_mark[Fullsize_2_mark == 0] = np.nan

        # Initialize new Registration variables
        newc = 0
        l = np.size(Registration1, axis=0)
        Registration2 = {}
        detector_old = {}
        detector_new = {}
        detector_numbering = {}
        c1 = 0
        c2 = 0
        c3 = 0
        c_numbering = 0
        cc = {}

        for i in range(stats2.shape[
                           0]):  # For each object in stats2 --------------------------------------------------------------------------------
            # print(f'Obj No. {i+1} (i)')
            max_object_intensity1 = 0
            max_object_intensity2 = 0
            b = stats2.coords[i]
            if time + 1 < 10:
                ttag = '00'
            elif time + 1 < 100:
                ttag = '0'
            else:
                ttag = ''

            threeDimg1 = niftiread(
                imageFolder + '/3DImage/' + imageNameOnly[:-4] + '/' + protein1Name + '/threeDimg_' +
                ttag + str(time + 1) + '.nii')
            # threeDimg2 = niftiread(
            #     imageFolder + '/3DImage/' + imageNameOnly[:-4] + '/' + protein2Name + '/threeDimg_' +
            #     ttag + str(time + 1) + '.nii')
            threeDimg2 = niftiread(
                imageFolder + '/3DImage/' + imageNameOnly[:-4] + '/' + protein1Name + '/threeDimg_' +
                ttag + str(time + 1) + '.nii')

            threeDimgPixelList1 = {}
            threeDimgPixelList2 = {}

            for i1 in range(np.size(b, axis=0)):
                threeDimgPixelList1[i1] = threeDimg1[b[i1, 0], b[i1, 1], b[i1, 2]]
                threeDimgPixelList2[i1] = threeDimg2[b[i1, 0], b[i1, 1], b[i1, 2]]

                if threeDimg1[b[i1, 0], b[i1, 1], b[i1, 2]] > max_object_intensity1:
                    max_object_intensity1 = threeDimg1[b[i1, 0], b[i1, 1], b[i1, 2]]

                if threeDimg2[b[i1, 0], b[i1, 1], b[i1, 2]] > max_object_intensity1:
                    max_object_intensity2 = threeDimg2[b[i1, 0], b[i1, 1], b[i1, 2]]

            threeDimgPixelList1 = sorted(threeDimgPixelList1, reverse=True)
            threeDimgPixelList2 = sorted(threeDimgPixelList2, reverse=True)

            # Average the pixels to get average object intensity
            average_object_intensity1 = sum(threeDimgPixelList1) / np.size(b, axis=0)
            average_object_intensity2 = sum(threeDimgPixelList2) / np.size(b, axis=0)

            a = {}
            a_t_1 = {}
            # k = boundary(b)                                                                         ### RRR

            for i1 in range(np.size(b, axis=0)):  # for each pixel in object b
                # print(b[i1, 0], b[i1, 1], b[i1, 2])
                # print(Fullsize_2_mark[b[i1, 0], b[i1, 1], b[i1, 2]])
                a[i1] = Fullsize_2_mark[b[i1, 0], b[i1, 1], b[
                    i1, 2]]  # add the value of that pixel in Fullsize_2_mark (i.e. the show-file cropped and mapped to t2 label file) to a

            datta = np.array(list(a.values()))
            value = statistics.mode(datta.flatten())

            value = np.nan if np.count_nonzero(np.isnan(datta)) > np.count_nonzero(datta == value) else value

            # if (np.isnan(value)):
            #     print('-------nan--------')

            if time > startpoint:  # Deal with fusion from previous time points
                for i1 in range(np.size(b, axis=0)):
                    a_t_1[i1] = Fullsize_1[b[i1, 0], b[i1, 1], b[i1, 2]]
                    # ----
                datta = np.array(list(a_t_1.values()))
                value_t_1 = statistics.mode(datta.flatten())
                Value_f_t_1 = np.count_nonzero(datta.flatten() == value_t_1)
                # Check whether the object has already merged in the last time point
                if not np.isnan(value_t_1) and isempty(intersect(value_t_1, np.array(Registration1[:, 0]))) \
                        and not isempty(
                    intersect(value_t_1, detector_fusion_old['detector3_fusion'])):  # merge happened in last time point
                    detector_numbering[c_numbering] = [value, value_t_1]
                    value = value_t_1
                    c_numbering = c_numbering + 1
                    # print(value)

            # print(f'Registration2 keys:')
            # print(np.array(list(Registration2.keys())))

            # Check whether the object has already been tracked in the current time point
            if not isempty(intersect(value, np.array(list(Registration2.values())))):
                value2 = setdiff(np.array(list(a.values())), np.array(list(Registration2.values()))[:, 0])

                if not isempty(value2) and np.size(value2) > 0 and not isempty(intersect(value2, Registration1[:, 0])):
                    value = value2[math.floor(rand() * np.size(value2))]

            if not np.isnan(value):
                value = int(value)

            # If the representatiove of an object is NaN, it means that it is a new object, Assign new ID to it

            if np.isnan(value):
                color = [0, 0, 0]
                newc += 1
                # print(f'new object {l + newc}')
                Registration2[l + newc] = [l + newc, stats2['centroid-0'][i], stats1['centroid-1'][i],
                                           stats1['centroid-2'][i]]
                value = l + newc
                # Reassign new labels to the sample
                for i1 in range(np.size(b, axis=0)):
                    Fullsize_2_mark[b[i1, 0], b[i1, 1], b[i1, 2]] = value
                    Fullsize_2_2[b[i1, 0], b[i1, 1], b[i1, 2]] = value

                txt = 'NEW ' + str(value)
                detector_new[c1] = (value)
                c1 += 1

                # Document the object characteristics
                xlswriter1.iloc[l+newc, time * 2 - 2] = 'new'
                xlswriter1.iloc[l+newc, time * 2 - 1] = l+newc

                xlswriter3.iloc[l+newc, time * 2 - 1] = max_object_intensity1
                xlswriter4.iloc[l+newc, time * 2 - 1] = average_object_intensity1
                xlswriter5.iloc[l+newc, time * 2 - 1] = np.size(b, axis=0)

                xlswriter6.iloc[l+newc, time * 2 - 1] = stats2['centroid-0'][i]
                xlswriter7.iloc[l+newc, time * 2 - 1] = stats2['centroid-1'][i]
                xlswriter8.iloc[l+newc, time * 2 - 1] = stats2['centroid-2'][i]

                xlswriter11.iloc[l+newc, time * 2 - 1] = max_object_intensity2
                xlswriter12.iloc[l+newc, time * 2 - 1] = average_object_intensity2

                # draw_text(value) = text(b(end, 1), b(end, 2), b(end, 3), txt, 'Rotation', +15)              # RRR

            # if the representative is not a NaN, it means we find a tracking
            elif not np.isnan(value) and value > 0:
                if isempty(intersect(value, np.array(
                        list(Registration2.keys())))):  # if value is already not in Registration 2,

                    Registration2[value] = [value, stats2['centroid-0'][i], stats2['centroid-1'][i],
                                            stats2['centroid-2'][i]]
                    # Reassign new labels to the sample
                    for i1 in range(np.size(b, axis=0)):
                        Fullsize_2_2[b[i1, 0], b[i1, 1], b[i1, 2]] = value

                    detector_old[c2] = value

                    # print(f'value {value} time {time}')    # -------------
                    c2 += 1
                    tx = time * 2 - 2
                    xlswriter1.iloc[value, time*2-2] = value
                    xlswriter1.iloc[value, time * 2 - 1] = value
                    xlswriter3.iloc[value, time * 2 - 1] = max_object_intensity1

                    xlswriter4.iloc[value, time * 2 - 1] = average_object_intensity1
                    xlswriter5.iloc[value, time * 2 - 1] = np.size(b, axis=0)

                    xlswriter6.iloc[value, time * 2 - 1] = stats2['centroid-0'][i]
                    xlswriter7.iloc[value, time * 2 - 1] = stats2['centroid-1'][i]
                    xlswriter8.iloc[value, time * 2 - 1] = stats2['centroid-2'][i]


                    xlswriter11.iloc[value, time * 2 - 1] = max_object_intensity2
                    xlswriter12.iloc[value, time * 2 - 1] = average_object_intensity2

                    draw_forsure = 0  #### RRR

                #
                # Draw code goes here. I'm confused.

                # If the representative is not NaN but is zero, it indicates a split
                # (Actually, if the correlation is there but the object id is already in Registration2)
                else:
                    # color = map(value,1:3)
                    newc += 1
                    Registration2[l + newc] = [l + newc, stats2['centroid-0'][i], stats2['centroid-1'][i],
                                               stats2['centroid-2'][i]]
                    detector_split[c3] = [value, l + newc]
                    c3 += 1
                    # print(f'split tracked {value} to {value} and {l+newc} at coordinates {Registration2[l+newc]}')

                    xlswriter10.iloc[value, time * 2 - 1] = value
                    xlswriter10.iloc[l+newc, time * 2 - 1] = value
                    # Reassign new labels to the sample
                    for i1 in range(np.size(b, axis=0)):
                        Fullsize_2_2[b[i1, 0], b[i1, 1], b[i1, 2]] = l + newc
                        Fullsize_2_mark[b[i1, 0], b[i1, 1], b[i1, 2]] = l + newc
                    for ix in range((time - 1) * 2):
                        if str(xlswriter1.iloc[value,ix])!="new":
                            xlswriter1.iloc[l+newc, ix] = xlswriter1.iloc[value, ix]

                    var = time * 2 - 1

                    xlswriter1.iloc[l+newc, time*2-2] = value
                    xlswriter1.iloc[l+newc, var] = l+newc
                    xlswriter3.iloc[l+newc, var] = max_object_intensity1
                    xlswriter4.iloc[l+newc, var] = average_object_intensity1
                    xlswriter5.iloc[l+newc, var] = np.size(b, axis=0)
                    xlswriter6.iloc[l+newc, var] = stats2['centroid-0'][i]
                    xlswriter7.iloc[l+newc, var] = stats2['centroid-1'][i]
                    xlswriter8.iloc[l+newc, var] = stats2['centroid-2'][i]
                    xlswriter11.iloc[l+newc, var] = max_object_intensity2
                    xlswriter12.iloc[l+newc, var] = average_object_intensity2

                    for i2 in range(time * 2):
                        # print(xlswriter1.iloc[value, i2])
                        if str(xlswriter1.iloc[value, i2]) != "new" and not pd.isna(xlswriter1.iloc[value, i2]):
                            value = str(xlswriter1.iloc[value, i2])
                            break
                    value = l + newc
        # print(xlswriter1)
        # print(xlswriter2)
        # colormap(map)             RRR
        # print(Registration2)
        newArr = np.zeros((max(sorted(Registration2)), 4))

        for pos in range(max(sorted(Registration2))):
            if pos in sorted(Registration2):
                newArr[pos, :] = Registration2[pos]
        # print(detector_split)
        # print(newArr)

        # Write the tracking result
        niftiwriteF(newArr, addr2 + 'Registration2_tracking_' + t2 + '.nii')
        niftiwriteF(Fullsize_2_2, addr2 + 'Fullsize_2_aftertracking_' + t2 + '.nii')

        # ====================== Tracking old and split object is almost done, now time for fusion detection and alarms =============================

        c = 0

        for i1 in range(stats2.shape[0]):  # for each object in stats2
            b = stats2.coords[i1]
            UNIQUEcount = {}

            for i2 in range(np.size(b, axis=0)):  # for each pixel in b
                # UNIQUEcount.append(Fullsize_2_mark[b[i2, 0], b[i2, 1], b[i2, 2]])
                UNIQUEcount[i2] = 0 if np.isnan(Fullsize_2_mark[b[i2, 0], b[i2, 1], b[i2, 2]]) else Fullsize_2_mark[
                    b[i2, 0], b[i2, 1], b[i2, 2]]

            uniq, cnts = np.unique(np.array(list(UNIQUEcount.values())), return_counts=True)
            uniq = uniq.astype(int)

            if len(uniq) > 1:  # if length(C) > 1
                detector_fusion[c] = uniq
                detector_fusion[c + 1] = cnts
                c += 2
        #
        # print(f'detector_fusion ------------------- before arrangement')
        # for ix in range(len(detector_fusion)):
        #     print(detector_fusion[ix])

        for i1 in range(0, len(detector_fusion), 2):
            if detector_fusion[i1][0] == 0:
                detector_fusion[i1 + 1] = detector_fusion[i1 + 1][1:]
                detector_fusion[i1] = detector_fusion[i1][1:]

        # print(f'detector_fusion after arrangement ============')
        # for ix in range(len(detector_fusion)):
        #     print(detector_fusion[ix])

        detector2_fusion = detector_fusion

        # Fusion alarm part 2
        for i1 in range(0, len(detector2_fusion), 2):
            for i2 in range(len(detector2_fusion[i1])):
                if not isempty(intersect(detector2_fusion[i1][i2], np.array(list(Registration2.values()))[:, 0])):
                    detector2_fusion[i1][i2] = 0
                    detector2_fusion[i1 + 1][i2] = 0
            for i2 in range(len(detector2_fusion[i1])):  # fusion size filter
                if detector2_fusion[i1 + 1][i2] < 5:
                    detector2_fusion[i1][i2] = 0
                    detector2_fusion[i1 + 1][i2] = 0

        # print(f'detector2_fusion ------------ after Fusion alarm and size filter')
        # for ix in range(len(detector2_fusion)):
        #     print(detector2_fusion[ix])

        c = 0

        for i1 in range(0, len(detector2_fusion), 2):
            if np.count_nonzero(detector2_fusion[i1]) != 0 and np.count_nonzero(detector_fusion[i1]) > 1:
                detector3_fusion[c] = detector_fusion[i1]
                detector3_fusion[c + 1] = detector_fusion[i1 + 1]
                # print(detector3_fusion[c])
                c += 2

        # print(f'detector3_fusion ----------------')
        for ix in range(0, len(detector3_fusion), 2):
            detector3_fusion[ix] = detector3_fusion[ix][detector3_fusion[ix] > 0]
            detector3_fusion[ix + 1] = detector3_fusion[ix + 1][detector3_fusion[ix + 1] > 0]
            # print(detector3_fusion[ix])
            # print(detector3_fusion[ix + 1])
            # print('----')
        print('Gathering results...')
        # print(f'detector_split =================')
        # for ix in range(len(detector_split)):
        #     print(detector_split[ix])
        ds_tosave = np.array(list(detector_split.values()), dtype=list)
        ds_tosave_mat = {"detector_split": ds_tosave}

        tempvar = 0
        sizes = {}
        varT = np.array(list(detector3_fusion.values()),dtype=list)
        for ixx in range(len(detector3_fusion)):
            if len(list(detector3_fusion.values())[ixx]) > tempvar:
                tempvar = len(list(detector3_fusion.values())[ixx])
                # print(f'tempvar {tempvar}')
            sizes[ixx] = len(list(detector3_fusion.values())[ixx])

        df_tosave = np.zeros((len(detector3_fusion), tempvar))
        for ix in range(np.size(df_tosave, axis=0)):
            for jx in range(np.size(varT[ix])):
                df_tosave[ix, jx] = varT[ix][jx]
        #
        # print(f'df_tosave ---------------------------')
        # print(df_tosave)
        df_tosave_mat = {"detector3_fusion": df_tosave}

        scio.savemat(addr2 + "fusion_tracking_" + t2 + ".mat", df_tosave_mat)
        scio.savemat(addr2 + "split_tracking_" + t2 + ".mat", ds_tosave_mat)

        xlswriter2.iloc[time, 0] = time+1
        xlswriter2.iloc[time, 1] = len(detector_old)
        xlswriter2.iloc[time, 2] = len(detector_new)
        xlswriter2.iloc[time, 3] = len(detector_split)
        xlswriter2.iloc[time, 4] = len(detector3_fusion)

        print(f'|  old  {len(detector_old)}  |  new  {len(detector_new)}  |  split {len(detector_split)}   |   merge  {len(detector3_fusion)}  |')

        print('Done.')


    # npz, pickle !!!!!!!!!!!!!!!

    xlswriter1 = xlswriter1.loc[2:,:]
    xlswriter2 = xlswriter2.iloc[1:,:]

    # writer = pd.ExcelWriter(folder + 'TEST' + str(timm) + '.xlsx', engine='xlsxwriter')
    # with pd.ExcelWriter(folder + 'TEST' + 'xxxxxxxxxx' + '.xlsx') as writer:
    workbook.close()

    print('Excel file created.')
    print(f'{excelFilename}')
    print('Saving to Excel.........')

    with pd.ExcelWriter(excelFilename, engine="openpyxl", mode="a", if_sheet_exists='overlay') as writer:

        xlswriter1.to_excel(writer, sheet_name='Sheet1', startrow=1, index=False, header=False)
        xlswriter2.to_excel(writer, sheet_name='Sheet2', startrow=1, index=False, header=False)
        xlswriter3.to_excel(writer, sheet_name='Sheet3', startrow=1, index=False, header=False)
        xlswriter4.to_excel(writer, sheet_name='Sheet4', startrow=1, index=False, header=False)
        xlswriter5.to_excel(writer, sheet_name='Sheet5', startrow=1, index=False, header=False)
        xlswriter6.to_excel(writer, sheet_name='Sheet6', startrow=1, index=False, header=False)
        xlswriter7.to_excel(writer, sheet_name='Sheet7', startrow=1, index=False, header=False)
        xlswriter8.to_excel(writer, sheet_name='Sheet8', startrow=1, index=False, header=False)
        xlswriter9.to_excel(writer, sheet_name='Sheet9', startrow=1, index=False, header=False)
        xlswriter10.to_excel(writer, sheet_name='Sheet10', startrow=1, index=False, header=False)
        xlswriter11.to_excel(writer, sheet_name='Sheet11', startrow=1, index=False, header=False)
        xlswriter12.to_excel(writer, sheet_name='Sheet12', startrow=1, index=False, header=False)
    # writer.save();
    starline()
    print('Step 2 Complete.')
    starline()

# USE DICTIONARY!!!!!!!!!!!!!!!!!!!!!
