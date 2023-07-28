# import numpy as np
# import pandas as pd
# from functions import niftiread,niftireadI, niftiwriteu16,size3, niftiwriteF
# from skimage import measure
# import statistics
#
# import statistics
#
# import scipy.io as scio
# import numpy as np
# import pandas as pd
# from functions import line
# from datetime import datetime
# import xlsxwriter
# import os
# import math
# import glob as glob
# import nibabel as nib
# from skimage import measure
# import re
# from correlation20220708 import correlation
# # from testCorr import correlation
# from functions import dashline, starline, niftiread, niftiwrite, niftiwriteF, intersect, setdiff, isempty, rand, nan_2d,niftireadI, niftiwriteu16
#
#
# # # time=1;t1='1';t2='2'
# # # trackedImageT1 = niftireadI('C:/Users/ramu_admin/Desktop/Baz/Apr6/SegTrack1014-1/1/Fullsize_label_1.nii')
# # # labeledImageT2 = niftireadI('C:/Users/ramu_admin/Desktop/Baz/Apr6/SegTrack1014-1/2/Fullsize_label_2.nii')
# # # maskT1 = niftireadI('C:/Users/ramu_admin/Desktop/Baz/Apr6/SegTrack1014-1/1/Fullsize_1.nii')
# # # maskT2 = niftireadI('C:/Users/ramu_admin/Desktop/Baz/Apr6/SegTrack1014-1/2/Fullsize_2.nii')
# # # weightsT1 = niftiread('C:/Users/ramu_admin/Desktop/Baz/Apr6/SegTrack1014-1/1/Weights_1.nii')
# # # weightsT2 = niftiread('C:/Users/ramu_admin/Desktop/Baz/Apr6/SegTrack1014-1/2/Weights_2.nii')
# # # padding=[20,20,2]
# # # addr = 'C:/Users/ramu_admin/Desktop/Baz/Apr6/SegTrack1014-1/'
#
# def correlationCalc(time,t1,t2,maskT1,maskT2,trackedImageT1, weightsT1,weightsT2,padding, addr,labeledT2):
#     time = time; t1 = t1; t2 = t2; maskT1 = maskT1; maskT2 = maskT2; weightsT1 = weightsT1; weightsT2 = weightsT2; padding = padding; addr = addr; trackedImageT1 = trackedImageT1
#     maskT1_pad = np.pad(maskT1, ((padding[0], padding[0]), (padding[1], padding[1]), (padding[2], padding[2])), 'constant')
#
#     # the regionprops_table function does not work with the mask itself, so have to use the 'labeled' image instead
#     trackedT1_pad = np.pad(trackedImageT1, ((padding[0], padding[0]), (padding[1], padding[1]), (padding[2], padding[2])),'constant')
#     maskT2_pad = np.pad(maskT1, ((padding[0], padding[0]), (padding[1], padding[1]), (padding[2], padding[2])), 'constant')
#     weightsT1_pad = np.pad(weightsT1, ((padding[0], padding[0]), (padding[1], padding[1]), (padding[2], padding[2]),(0,0)), 'constant')
#     weightsT2_pad = np.pad(weightsT2, ((padding[0], padding[0]), (padding[1], padding[1]), (padding[2], padding[2]),(0,0)), 'constant')
#
#     labeledT2_pad = np.pad(labeledT2, ((padding[0], padding[0]), (padding[1], padding[1]), (padding[2], padding[2])),'constant')
#
#     [x, y, z] = size3(maskT1)
#     correlation_corr = np.zeros((x + padding[0] * 2, y + padding[1] * 2, z + padding[2] * 2))
#     correlation_id = np.zeros((x + padding[0] * 2, y + padding[1] * 2, z + padding[2] * 2))
#
#     trackedT1 = trackedT1_pad
#
#     # get the label and coords of each regionprop (or an object)
#     stats1 = pd.DataFrame(measure.regionprops_table(labeledT2_pad, properties=('label', 'coords')))
#     VoxelList = stats1.coords # List of all pixels in each regionprop
#
#     for i in range(0, stats1.shape[0]): # for each object
#         objSize = np.size(VoxelList[i],axis=0)
#
#         if objSize < 30:
#             stepsize = 1
#         else:
#             stepsize=1 # to speed-up the process, we take every third pixel as the center of the window
#
#         # # turns out we don't need this
#         # extremes=[[np.min(VoxelList[i][:,0]),np.max(VoxelList[i][:,0])],[np.min(VoxelList[i][:,1]),np.max(VoxelList[i][:,1])],[np.min(VoxelList[i][:,2]),np.max(VoxelList[i][:,2])]]
#
#         for index in range(0,objSize,stepsize):
#             Feature_map2 = np.copy(weightsT2_pad[
#                                    VoxelList[i][index][0] - 3:VoxelList[i][index][0] + 3 + 1,
#                                    VoxelList[i][index][1] - 3:VoxelList[i][index][1] + 3 + 1,
#                                    VoxelList[i][index][2] - 1:VoxelList[i][index][2] + 1 + 1,
#                                    :])
#             for x in [-2,-1,0,1,2]:
#                 for y in [-2,-1,0,1,2]:
#                     for z in [ -1,0,1]:
#
#                         Feature_map1 = np.copy(weightsT1_pad[
#                                 VoxelList[i][index][0] + x - 3:VoxelList[i][index][0] + x + 3 + 1,
#                                 VoxelList[i][index][1] + y - 3:VoxelList[i][index][1] + y + 3 + 1,
#                                 VoxelList[i][index][2] + z - 1:VoxelList[i][index][2] + z + 1 + 1,
#                                 :])
#                         # # Flattening the feature map
#                         Feature_map1_flatten = Feature_map1.flatten(order='F')
#                         Feature_map2_flatten = Feature_map2.flatten(order='F')
#
#                         # calculate correlation
#                         corr = np.corrcoef(Feature_map2_flatten,Feature_map1_flatten)[0, 1]
#                         print(corr)
#                         if corr > 0.2:
#                             if correlation_corr[VoxelList[i][index][0] + x, VoxelList[i][index][1] + y, VoxelList[i][index][ 2] + z] < corr:
#                                 b = VoxelList[i]
#                                 a = []
#                                 for i1 in range(len(b)):
#                                     a.append(trackedT1[b[i1][0],b[i1][1],b[i1][2]])
#
#                                 value = statistics.mode(np.array(a).flatten())
#                                 u, c = np.unique(np.array(a), return_counts=True)
#                                 try:
#                                     countZero = dict(zip(u, c))[0]
#                                 except:
#                                     countZero = 0
#
#                                 if countZero>value:
#                                     value=0
#
#                                 # correlation_id[VoxelList[i][index][0] - 3 : VoxelList[i][index][0] + 4, VoxelList[i][index][1] - 3 : VoxelList[i][index][1] + 4, VoxelList[i][index][2] -1 : VoxelList[i][index][2] + 2] = value
#
#                                 ####
#
#     niftiwriteu16(correlation_id, 'C:/Users/ramu_admin/Desktop/Baz/Apr6/SegTrack1014-1/2/__XXXXXXXXXXXXXXXXXXXXXXXX_ID.nii')
#
#     cropped_correlation_id = correlation_id * maskT2_pad
#
#     niftiwriteu16(cropped_correlation_id,
#                   'C:/Users/ramu_admin/Desktop/Baz/Apr6/SegTrack1014-1/2/__XXXXXXXXXXXXXXXXXXXXXXXX_ID_CROPPPPPPPP.nii')
#     niftiwriteF(correlation_corr,
#                   'C:/Users/ramu_admin/Desktop/Baz/Apr6/SegTrack1014-1/2/__CCCCCCCCCCCCCCCCCC_CORR.nii')
#
# # correlationCalc(time,t1,t2,maskT1,maskT2,trackedImageT1, weightsT1,weightsT2,padding, addr,labeledImageT2)

########################################################################################################################

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
from functions import dashline, starline, niftiread, niftiwrite, niftiwriteF, intersect, setdiff, isempty, rand, nan_2d,niftireadI, niftiwriteu16,niftireadu32, niftiwriteu32
from createEventIntensityPlots import createEventsAndIntensityPlots
def myTrackStep2(seg_op_folder, track_op_folder, imageNameS, imageNameO, protein1Name, protein2Name, modelName='FC-DenseNet', initialpoint=1, startpoint=1, endpoint=40, trackbackT=2):
    protein1Name = protein1Name
    protein2Name = protein2Name
    imageName=imageNameS
    imageFolder = os.path.dirname(imageName)
    imageNameOnly = os.path.basename(imageName)
    starline()  # print **************************************
    print('                     step 2 begin')
    starline()
    # colormap = scio.loadmat('/home/nirvan/Desktop/Projects/MATLAB CODES/colormap.mat')
    im = niftiread(imageName)
    sz = np.shape(im)
    I3dw = [sz[0], sz[1], sz[2]]
    # I3dw = [512, 280, 15]
    padding = [20, 20, 2]
    timm = datetime.now()

    folder = track_op_folder
    # trackbackT = 2

    if not os.path.isdir(folder):
        print(os.makedirs(folder))

    depth = 64  # the deep features to take in correlation calculation

    print(f'depth = {depth}, startpoint = {startpoint}, endpoint = {endpoint}')

    spatial_extend_matrix = np.full((10, 10, 3, depth),
                                    0)  # the weight decay of 'extended search' (not used right now in correlation calculation)

    for i1 in range(0, 10):
        for i2 in range(0, 10):
            for i3 in range(0, 3):
                spatial_extend_matrix[i1, i2, i3, :] = math.exp(((i1 + 1 - 5) + (i2 + 1 - 5) + (i3 + 1 - 2)) / 20)

    print(folder)
    globalSplitList = []
    globalMergeList = []
    globalTrackList = []
    globalDeathList = []
    globalBirthList = []
    globalIdList = []

    globalTargetIdList=[]

    for time in range(startpoint, endpoint + 1):
        dashline()
        tic = datetime.now()
        t1 = str(time)
        t2 = str(time + 1)

        print(f'time point: {t1} --> {t2}')

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
        if time == startpoint:
            # for i1 in range(1, time - initialpoint + 1 + 1):
            i1=1
            Fullsize_2 = niftireadu32(addr2 + 'Fullsize_label_' + t2 + '.nii')
            Fullsize_regression_2 = niftiread(addr2 + 'Weights_' + t2 + '.nii')
            try:
                Fullsize_1 = niftireadu32(addr1 + 'Fullsize_2_aftertracking_' + t1 + '.nii')
                Fullsize_regression_1 = niftiread(addr1 + 'Weights_' + t1 + '.nii')

            except FileNotFoundError:
                Fullsize_1 = niftireadu32(addr1 + 'Fullsize_label_' + t1 + '.nii')
                Fullsize_regression_1 = niftiread(addr1 + 'Weights_' + t1 + '.nii')

            correlation(Fullsize_1, Fullsize_2, Fullsize_regression_1, Fullsize_regression_2, t2, i1,
                        spatial_extend_matrix, addr2, padding)

        else:
            i1=1
            Fullsize_2 = niftireadu32(addr2 + 'Fullsize_label_' + t2 + '.nii')
            Fullsize_regression_2 = niftiread(addr2 + 'Weights_' + t2 + '.nii')
            Fullsize_1 = niftireadu32(addr1 + 'Fullsize_2_aftertracking_' + t1 + '.nii')
            Fullsize_regression_1 = niftiread(addr1 + 'Weights_' + t1 + '.nii')
            correlation(Fullsize_1, Fullsize_2, Fullsize_regression_1, Fullsize_regression_2, t2, i1,
                        spatial_extend_matrix, addr2, padding)


        t1 = str(time)
        t2 = str(time + 1)

        if time == initialpoint:
            labeled1 = niftireadu32(folder + t1 + '/Fullsize_label_' + t1 + '.nii')
        else:
            labeled1 = niftireadu32(folder + t1 + '/Fullsize_2_aftertracking_' + t1 + '.nii')

        idlist_previous = []

        stats_t1 = pd.DataFrame(measure.regionprops_table(labeled1, properties=('label', 'coords')))

        for obj1 in range(len(stats_t1)):
            idlist_previous.append(
                labeled1[(stats_t1.coords[obj1][0][0], stats_t1.coords[obj1][0][1], stats_t1.coords[obj1][0][2])])

        if time == startpoint:
            globalIdList.append(idlist_previous)
            for igil in range(len(idlist_previous)):
                globalTargetIdList.append(idlist_previous[igil])

        maxx = np.amax(labeled1)

        labeled2 = niftireadu32(folder + t2 + '/Fullsize_label_' + t2 + '.nii')
        corr_id_2 = niftireadu32(folder + t2 + '/correlation_map_padding_show_traceback1_'+t2+'.nii')

        corr2_crop = corr_id_2[padding[0]:sz[0] + padding[0], padding[1]:sz[1] + padding[1],
                     padding[2]:sz[2] + padding[2]]
        corr2_crop[labeled2 == 0] = 0

        newArray = corr2_crop.copy()
        statsfullsize2 = pd.DataFrame(measure.regionprops_table(labeled2, properties=('label', 'coords')))
        count = 0
        retiredList = []
        mergedList = []
        newList = []
        for obj in range(len(statsfullsize2)):
            idlist = []
            for i in statsfullsize2.coords[obj]:
                idlist.append(corr2_crop[(i[0], i[1], i[2])])
            mod = statistics.mode(idlist)
            if mod == 0:
                for i in statsfullsize2.coords[obj]:
                    newArray[(i[0], i[1], i[2])] = 0

            if len(np.unique(idlist)) > 1 and mod != 0:
                for i in statsfullsize2.coords[obj]:
                    if corr2_crop[(i[0], i[1], i[2])] == 0 or corr2_crop[(i[0], i[1], i[2])] == mod:
                        newArray[(i[0], i[1], i[2])] = mod

        # starline()
        # print("mode zero set all to zero, zero set to mode complete.")
        #
        # print(maxx)

        newcount = 0
        for obj in range(len(statsfullsize2)):
            idlist = []
            for i in statsfullsize2.coords[obj]:
                idlist.append(newArray[(i[0], i[1], i[2])])
            mod = statistics.mode(idlist)
            if mod == 0:
                newcount += 1
                for i in statsfullsize2.coords[obj]:
                    newArray[(i[0], i[1], i[2])] = maxx + 1
                newList.append(maxx + 1)
                globalTargetIdList.append(maxx+1)
                maxx += 1

            if len(np.unique(idlist)) > 1:
                u, c = np.unique(np.array(idlist), return_counts=True)

                for count_index, count_i in enumerate(c):
                    if count_i < 3:
                        for i in statsfullsize2.coords[obj]:
                            if newArray[(i[0], i[1], i[2])] == u[count_index]:
                                newArray[(i[0], i[1], i[2])] = mod
        # print("all new objects are given a new id.")
        # print("all tiny merges filtered out")

        # print(maxx)

        # add to mergelist
        for obj in range(len(statsfullsize2)):
            idlist = []
            for i in statsfullsize2.coords[obj]:
                idlist.append(newArray[(i[0], i[1], i[2])])
            if len(np.unique(idlist)) > 1:
                mod = statistics.mode(idlist)
                u, c = np.unique(np.array(idlist), return_counts=True)
                # print(u, c)
                for uniqueIndex, uniqueCount in enumerate(c):
                    if u[uniqueIndex] != mod:
                        # print(f"{mod} swallowed {u[uniqueIndex]} at time {time + 1}")
                        mergedList.append([mod, u[uniqueIndex], time + 1])
                        for i in statsfullsize2.coords[obj]:
                            newArray[(i[0], i[1], i[2])] = mod

        # print(mergedList)

        for obj in range(len(statsfullsize2)):
            idlist = []
            for i in statsfullsize2.coords[obj]:
                idlist.append(newArray[(i[0], i[1], i[2])])

        # take care of the split
        accountedSplitIdList = []
        splitList = []
        for obj in range(len(statsfullsize2)):
            idlist = []
            for i in statsfullsize2.coords[obj]:
                idlist.append(newArray[(i[0], i[1], i[2])])
            u, c = np.unique(idlist, return_counts=True)
            mode_i = statistics.mode(idlist)
            if mode_i not in accountedSplitIdList:
                accountedSplitIdList.append(mode_i)
            else:
                for j in statsfullsize2.coords[obj]:
                    newArray[(j[0], j[1], j[2])] = maxx + 1
                splitList.append([mode_i, maxx + 1, time + 1])
                # print(f'{mode_i} splitted into {mode_i} and {maxx + 1} at t={time + 1}')
                maxx += 1

        # print(splitList)

        idlist_now = []
        deathList = []
        for obj2 in range(len(statsfullsize2)):
            idlist_now.append(newArray[(statsfullsize2.coords[obj2][0][0], statsfullsize2.coords[obj2][0][1],
                                        statsfullsize2.coords[obj2][0][2])])

        newTotal = len(idlist_now)
        oldTotal = len(idlist_previous)

        trackedList = []
        newList = []
        for theId in idlist_now:
            if theId in idlist_previous:
                trackedList.append([theId, time + 1])
            else:
                newList.append([theId, time + 1])

        for oldId in idlist_previous:
            if oldId not in idlist_now:
                deathList.append([oldId, time + 1])

        # print(trackedList)
        # print(newList)

        # np.array(newList).toFile(folder + t2 + '/newList' + t2 + '.csv', sep=',')
        # np.array(trackedList).toFile(folder + t2 + '/trackedList' + t2 + '.csv', sep=',')
        # np.array(deathList).toFile(folder + t2 + '/deathList' + t2 + '.csv', sep=',')
        # np.array(mergedList).toFile(folder + t2 + '/mergedList' + t2 + '.csv', sep=',')
        # np.array(splitList).tofile(folder + t2 + '/splitList' + t2 + '.csv', sep=',')
        #
        # newListDF = pd.DataFrame(newList, columns=['merge_time', 'merged', 'merged_into', 'merged_id_starttime'])

        globalSplitList.append(splitList)
        globalMergeList.append(mergedList)
        globalTrackList.append(trackedList)
        globalDeathList.append(deathList)
        globalBirthList.append(newList)
        globalIdList.append(idlist_now)

        niftiwriteu32(newArray, folder + t2 + '/Fullsize_2_aftertracking_' + t2 + '.nii')

        max_old=maxx

    # np.array(globalSplitList).toFile(folder + '/globalSplitList.csv', sep=',')
    # np.array(globalMergeList).toFile(folder + '/globalMergeList.csv', sep=',')
    # np.array(globalTrackList).toFile(folder + '/globalTrackList.csv', sep=',')
    # np.array(globalDeathList).toFile(folder + '/globalDeathList.csv', sep=',')
    # np.array(globalBirthList).tofile(folder + '/globalBirthList.csv', sep=',')
    # starline()
    # starline()
    # print(globalMergeList)
    # # dashline()
    # print(globalSplitList)
    # # dashline()
    # print(globalDeathList)
    # # dashline()
    # print(globalBirthList)
    # # dashline()
    # print(globalIdList)

    maxid = np.amax(globalIdList[len(globalIdList) - 1])


    ################################################################ Combining Tracking Results into a single file ####
    starline()
    print('Combining tracking results.....')

    tempMat = niftireadu32(track_op_folder + '1/Fullsize_label_1.nii')
    x, y, z = np.shape(tempMat)
    finalMatrix = np.zeros(shape=(x, y, z, endpoint - startpoint + 2))
    # print(np.shape(finalMatrix))
    for timepoint in range(startpoint, endpoint + 2):
        if timepoint == 1:
            tMatrix = niftireadu32(track_op_folder + str(timepoint) + '/Fullsize_label_1.nii')
        else:
            tMatrix = niftireadu32(
                track_op_folder + str(timepoint) + '/Fullsize_2_aftertracking_' + str(timepoint) + '.nii')

        finalMatrix[:, :, :, timepoint - 1] = tMatrix

    niftiwriteu32(finalMatrix, track_op_folder + 'TrackedCombined.nii')


    # excelFilename = folder + 'TrackingID' + re.sub(r'\W+', '_', str(timm)) + '.xlsx'  # the excel file name to write the tracking result
    #
    # workbook = xlsxwriter.Workbook(excelFilename)
    # worksheet1 = workbook.add_worksheet()
    # worksheet2 = workbook.add_worksheet()
    # worksheet3 = workbook.add_worksheet()
    #
    # xlsxwriter1 = pd.DataFrame(nan_2d(maxid+1,endpoint*2))
    # xlsxwriter2 = pd.DataFrame(np.zeros(endpoint,4))
    # xlsxwriter3 = pd.DataFrame(nan_2d(maxid+1,endpoint*2))

    # worksheet2.write('A1', 'time')  # write titles to excel
    # worksheet2.write('B1', 'old')
    # worksheet2.write('C1', 'new')
    # worksheet2.write('D1', 'split')
    # worksheet2.write('E1', 'fusion')

    globalSplitList = [ [splitEvent for splitEvent in timeSplitList if splitEvent[0] != 1] for timeSplitList in globalSplitList ]

    print(globalIdList)
    lst = []
    for i in range(1, len(globalIdList) + 1):
        if i == 1:
            lst.append(str(i))
        else:
            lst.append(str(i))
            lst.append(str(i))
    mydf = pd.DataFrame(nan_2d(100, len(globalIdList) * 2 - 1))
    # print(mydf.head())
    parent = 0
    print(maxid)
    # print(len(globalIdList))
    for idd in range(2, maxid):
        if idd%100==0:
            if idd%3000==0:
                print('#',end='\n')
            print('#',end='')
        for tt in range(len(globalSplitList) + 1):
            if idd in globalIdList[tt]:
                if tt == 0:
                    mydf.loc[idd - 2, tt] = int(idd)
                else:
                    mydf.loc[idd - 2, tt * 2 - 1] = int(idd)
                    mydf.loc[idd - 2, tt * 2] = int(idd)
                    if idd not in globalIdList[tt - 1]:  # if the object id is new
                        # if idd in (np.array(globalSplitList[tt - 1])[:, 0:2].flatten()):  #replaced with next line
                        # if the object split from another id
                        # list comprehension removes the empty sublist so that [:,0:2] doesn't run into an error
                        # temporary__ =
                        if idd in (np.array([sublist for sublist in globalSplitList[tt - 1] if sublist])[:, 0:2].flatten()):
                            for splitEvent in globalSplitList[tt - 1]:  # for all split events at that time
                                if splitEvent[1] == idd:  # if idd in an event
                                    parent = splitEvent[0]  # get parent id
                                    for parent_t in range(tt):  # for all times until tt
                                        if tt == 1:
                                            if not pd.isna(mydf.loc[parent - 2, parent_t]):
                                                mydf.loc[idd - 2, parent_t] = mydf.loc[parent - 2, parent_t]
                                        if tt > 1:
                                            if parent_t == 0 and not pd.isna(mydf.loc[parent - 2, parent_t]) and not mydf.iloc[parent - 2, parent_t] == 'new':
                                                mydf.loc[idd - 2, parent_t] = mydf.loc[parent - 2, parent_t]
                                            if parent_t > 0 and not pd.isna(
                                                    mydf.loc[parent - 2, parent_t * 2 - 1]) and not mydf.iloc[
                                                                                                        parent - 2, parent_t * 2 - 1] == 'new':
                                                mydf.loc[idd - 2, parent_t * 2 - 1] = mydf.loc[
                                                    parent - 2, parent_t * 2 - 1]
                                            if parent_t > 0 and not pd.isna(mydf.loc[parent - 2, parent_t * 2]) and not \
                                            mydf.iloc[parent - 2, parent_t * 2] == 'new':
                                                mydf.loc[idd - 2, parent_t * 2] = mydf.loc[parent - 2, parent_t * 2]
                        else:
                            mydf.loc[idd - 2, tt * 2 - 2] = 'new'

    mydf.columns = lst

    mydf.to_csv(folder+'tracking_result.csv',index=False)

    print('Generating Events list..')
    # get the count of events
    globalEventCountList=np.zeros((len(globalSplitList)+1,7))
    for i in range(len(globalSplitList)+1):

        if i == 0:
            globalEventCountList[i, 0] = len(globalIdList[i])
            globalEventCountList[i, 1] = 0
            globalEventCountList[i, 2] = 0
            globalEventCountList[i, 3] = 0
            globalEventCountList[i, 4] = 0
            globalEventCountList[i, 5] = 0
            globalEventCountList[i, 6] = 0

        else:
            globalEventCountList[i, 0] = len(globalIdList[i]) #obj count
            globalEventCountList[i, 1] = len(globalSplitList[i-1]) #split count
            globalEventCountList[i, 2] = len(globalMergeList[i-1]) #merge count
            globalEventCountList[i, 3] = len(globalBirthList[i-1]) #count of ids that were new
            globalEventCountList[i, 4] = len(globalDeathList[i-1]) #count of ids that were no longer continued
            globalEventCountList[i, 5] = len(globalBirthList[i-1])-len(globalSplitList[i-1]) # entirely new objects
            globalEventCountList[i, 6] = len(globalDeathList[i - 1]) - len(globalMergeList[i - 1])  # entirely dead objects

    eventCols=['Total objects','Split','Merge','new id','retired id','birth','death']
    eventDF = pd.DataFrame(globalEventCountList)
    eventDF.columns = eventCols

    eventDF.to_csv(folder + 'all_events.csv', index=False)

    # this part suffers from redundancy. Clear this up
    tmplst = []
    for l in range(len(globalMergeList)):
        for m in range(len(globalMergeList[l])):
            tmplst.append(globalMergeList[l][m])
    mergeDF = pd.DataFrame(np.array(tmplst))
    mergeCols = ['Merged Into','Merged','Time']
    mergeDF.columns = mergeCols
    mergeDF.to_csv(folder + 'merge_list.csv', index=False)

    # this part also suffers from redundancy. Clear this up
    tmplst2 = []
    for l in range(len(globalSplitList)):
        for m in range(len(globalSplitList[l])):
            tmplst2.append(globalSplitList[l][m])
    splitDF = pd.DataFrame(np.array(tmplst2))
    splitCols = ['Splitted', 'Splitted Into', 'Time']
    splitDF.columns = splitCols
    splitDF.to_csv(folder + 'split_list.csv', index=False)


    ################################################################

    #DataFrameM titles
    lst = []
    for i in range(1, len(globalMergeList) + 2):
        if i == 1 or i == len(globalMergeList) + 1:
            lst.append(str(i))
        else:
            lst.append(str(i))
            lst.append(str(i))

    # DataFrameS&M titles
    lst2 = []
    for i in range(1, len(globalMergeList) + 2):
        if i == 1:
            lst2.append(str(i))
        else:
            lst2.append(str(i) + '_split')
            lst2.append(str(i) + '_merge')

    allMerges = [item for sublist in globalMergeList for item in sublist]
    mrgDF = pd.DataFrame(nan_2d(maxid, len(globalMergeList) * 2 - 1))
    combinedDF = pd.DataFrame(nan_2d(maxid, len(globalMergeList) * 2 - 1))

    for eachMerge in allMerges:
        lion = eachMerge[0]
        deer = eachMerge[1]
        clock = eachMerge[2]
        mrgDF.loc[lion - 2, (clock - 1) * 2] = deer
        combinedDF.loc[lion - 2, (clock - 1) * 2] = deer

    mrgDF.columns = lst
    print('Saving Merge Events...')
    mrgDF.to_csv(folder + 'Sheet10.csv', index=False)

    ################################################################

    allSplits = [item for sublist in globalSplitList for item in sublist]
    lst = []
    for i in range(1, len(globalSplitList) + 2):
        if i == 1 or i == len(globalSplitList) + 1:
            lst.append(str(i))
        else:
            lst.append(str(i))
            lst.append(str(i))
    sptDF = pd.DataFrame(nan_2d(maxid, len(globalSplitList) * 2 - 1))

    for eachSplit in allSplits:
        parentt = eachSplit[0]
        childd = eachSplit[1]
        hourr = eachSplit[2]
        sptDF.loc[parentt - 2, (hourr - 1) * 2 - 1] = childd
        combinedDF.loc[parentt - 2, (hourr - 1) * 2 - 1] = childd

    sptDF.columns = lst
    print('Saving Split Events...')
    sptDF.to_csv(folder + 'Sheet11.csv', index=False)

    print('Saving Combined Events...')
    combinedDF.columns = lst2
    combinedDF.to_csv(folder+'Sheet12.csv',index=False)

    ################################################################

    import itertools

    flattened_list = list(itertools.chain(*globalIdList))
    unique_values = set(flattened_list)
    allIds = list(unique_values)

    allSplits = np.array(allSplits)
    allChild = allSplits[:, 1]
    allParents = allSplits[:, 0]

    globalIdExistForArray = np.zeros((len(allIds), 4))

    for anId in allIds:
        globalIdExistForArray[anId - 1, 0] = anId
        existCount = 0
        idFirstExist = 0
        idLastExist = 0
        tx = 0
        for timeIdList in globalIdList:

            if anId in timeIdList and globalIdExistForArray[anId - 1, 1] == 0:
                globalIdExistForArray[anId - 1, 1] = anId
                idFirstExist = tx + 1
                idLastExist = tx + 1
            elif anId in timeIdList:
                idLastExist += 1

            globalIdExistForArray[anId - 1, 1] = idFirstExist
            globalIdExistForArray[anId - 1, 2] = idLastExist

            if anId in allChild:
                parent = allParents[allChild == anId]
                globalIdExistForArray[anId - 1, 3] = parent
            else:
                globalIdExistForArray[anId - 1, 3] = anId

            tx += 1

    globalIdExistForArray = np.uint16(globalIdExistForArray)
    # print(globalIdExistForArray[320:370])

    globalIdExistForDF = pd.DataFrame(globalIdExistForArray)
    globalIdExistForDF.columns = ['index','timestart','timeend','parent']

    print('Saving when the objects exist...')
    globalIdExistForDF.to_csv(folder + 'Sheet3.csv', index=False)


    print('Saving all possible target IDs...')
    globalTargetIdListDF = pd.DataFrame(globalTargetIdList)
    globalTargetIdListDF.columns=['targetId']
    globalTargetIdListDF.to_csv(folder + 'target_IDs.csv', index=False)

    print('Generating events and intensity plots...')
    print(track_op_folder)
    print(imageFolder)
    print(imageNameO)

    spDF = pd.read_csv(seg_op_folder+'segmentation_parameters.csv')
    sT = spDF['startTime'].loc[0]-1
    eT = spDF['endTime'].loc[0]

    createEventsAndIntensityPlots(segpath=seg_op_folder, filePath=track_op_folder, modelName=modelName, originalImage=imageNameO, sT=sT, eT=eT)
    # createEventsAndIntensityPlots(filePath=track_op_folder, originalImage=imageFolder + '/' + imageNameOnly.split('.')[0]+'.tif', nameOnly=imageNameOnly.split('.')[0], distance=endpoint-startpoint)






