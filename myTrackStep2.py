import statistics
import json
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
import matplotlib.pyplot as plt
import re
from correlation20220708 import correlation
# from testCorr import correlation
from functions import dashline, starline, niftiread, niftiwrite, niftiwriteF, intersect, setdiff, isempty, rand, nan_2d, \
    niftireadI, niftiwriteu16, niftireadu32, niftiwriteu32
from createEventIntensityPlots import runAnalysis, runAnalysisNewWay
    # createEventsAndIntensityPlots
import time as theTime


def myTrackStep2(seg_op_folder, track_op_folder, imageNameS, imageNameO, protein1Name, protein2Name,
                 modelName='FC-DenseNet', initialpoint=1, startpoint=1, endpoint=40, trackbackT=2,
                 stime=theTime.perf_counter()):
    # sT = 16; endpoint = endpoint
    # runAnalysisNewWay(origImgPath=imageNameO,
    #                   trackedimagepath=track_op_folder + 'TrackedCombined.nii',
    #                   sT=sT, eT=sT + endpoint,
    #                   plotsavepath=track_op_folder[:-1])

    protein1Name = protein1Name
    protein2Name = protein2Name
    initialpoint = startpoint
    imageName = imageNameS
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

    globalSplitFilterList = []
    globalMergeFilterList = []

    globalTrueMergeList = []
    globalSplitAndMergeList = []

    globalTargetIdList = []

    birth_id_dict = {}
    merged_id_dict = {}
    truemerged_id_dict = {}
    splitmerge_id_dict = {}
    split_id_dict = {}
    id_that_split_dict = {}
    dead_id_dict = {}
    all_id_dict = {}
    event_count_dict = {}
    truemerge_as_primary_id_dict = {}
    truemerge_as_secondary_id_dict = {}

    events_counts = np.zeros((endpoint-startpoint+2,7))

    maxx = 0

    for time in range(startpoint, endpoint + 1):
        dashline()
        tic = datetime.now()
        t1 = str(time)
        t2 = str(time + 1)

        print(f'time point: {t1} --> {t2}')

        addr1 = folder + t1 + '/'
        addr2 = folder + t2 + '/'

        Files1 = sorted(glob.glob(addr1 + '*.nii'))
        Files2 = sorted(glob.glob(addr2 + '.nii'))

        # calculate correlation between this and next time point, using (labeled images and weights from step 1)

        # if time - initialpoint < trackbackT:  # calculating correlation for start time points (e.g. time=2)
        if time == startpoint:
            # for i1 in range(1, time - initialpoint + 1 + 1):
            i1 = 1
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
            i1 = 1
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
            all_id_dict[time] =sorted([int(identity) for identity in idlist_previous])
            birth_id_dict[time]=sorted([int(identity) for identity in idlist_previous])

            maxx = np.amax(labeled1)
            # print(all_id_dict[time], birth_id_dict[time])
            events_counts[0,:]=[len(all_id_dict[time]),0,0,len(birth_id_dict[time]),0,0,0]
            for igil in range(len(idlist_previous)):
                globalTargetIdList.append(idlist_previous[igil])

        labeled2 = niftireadu32(folder + t2 + '/Fullsize_label_' + t2 + '.nii')
        corr_id_2 = niftireadu32(folder + t2 + '/correlation_map_padding_show_traceback1_' + t2 + '.nii')

        corr2_crop = corr_id_2[padding[0]:sz[0] + padding[0], padding[1]:sz[1] + padding[1],
                     padding[2]:sz[2] + padding[2]]
        corr2_crop[labeled2 == 0] = 0

        newArray = corr2_crop.copy()
        statsfullsize2 = pd.DataFrame(measure.regionprops_table(labeled2, properties=('label', 'coords')))
        count = 0

        mergedList = []
        trueMergedList = []
        splitMergedList = []
        newList = []
        bornlist = []

############################################################################################################################################################
        # For each object
        for obj in range(len(statsfullsize2)):
            idlist = []
            for i in statsfullsize2.coords[obj]:
                idlist.append(corr2_crop[(i[0], i[1], i[2])])
            mod = statistics.mode(idlist)
            # If mode is zero, set all pixels to zero
            if mod == 0:
                for i in statsfullsize2.coords[obj]:
                    newArray[(i[0], i[1], i[2])] = 0
            # if mode is not zero, set all zero pixels to mode
            if len(np.unique(idlist)) > 1 and mod != 0:
                for i in statsfullsize2.coords[obj]:
                    if corr2_crop[(i[0], i[1], i[2])] == 0 or corr2_crop[(i[0], i[1], i[2])] == mod:
                        newArray[(i[0], i[1], i[2])] = mod

        newcount = 0
        # for each object
        for obj in range(len(statsfullsize2)):
            idlist = []
            for i in statsfullsize2.coords[obj]:
                idlist.append(newArray[(i[0], i[1], i[2])])
            mod = statistics.mode(idlist)

            # if mode is still zero
            if mod == 0:
                newcount += 1
                # set every pixel to a new id
                for i in statsfullsize2.coords[obj]:
                    newArray[(i[0], i[1], i[2])] = maxx + 1
                newList.append(maxx + 1)
                bornlist.append(maxx+1)
                globalTargetIdList.append(maxx + 1)
                maxx += 1

            # if still more than one id
            if len(np.unique(idlist)) > 1:
                u, c = np.unique(np.array(idlist), return_counts=True)

                for count_index, count_i in enumerate(c):
                    # replace ids existing for less than 3 pixels with mode
                    if count_i < 3:
                        for i in statsfullsize2.coords[obj]:
                            if newArray[(i[0], i[1], i[2])] == u[count_index]:
                                newArray[(i[0], i[1], i[2])] = mod

        # birth_id_dict[time+1] = sorted([int(identity) for identity in bornlist]) # moved below filter

        merged_id_list=[]
        truemerged_id_list=[]
        # add to mergelist
        for obj in range(len(statsfullsize2)): # for all  objects
            idlist = []
            for i in statsfullsize2.coords[obj]:
                idlist.append(newArray[(i[0], i[1], i[2])])
            if len(np.unique(idlist)) > 1: # if more than one ID
                mod = statistics.mode(idlist)
                u, c = np.unique(np.array(idlist), return_counts=True)
                # print(u, c)
                for uniqueIndex, uniqueCount in enumerate(c):
                    if u[uniqueIndex] != mod:
                        # print(f"{mod} swallowed {u[uniqueIndex]} at time {time + 1}")
                        mergedList.append([mod, u[uniqueIndex], time + 1])
                        merged_id_list.append(u[uniqueIndex])
                        for i in statsfullsize2.coords[obj]:
                            newArray[(i[0], i[1], i[2])] = mod

        for obj in range(len(statsfullsize2)):
            idlist = []
            for i in statsfullsize2.coords[obj]:
                idlist.append(newArray[(i[0], i[1], i[2])])

        # take care of the split
        splitIdList = []
        idThatSplitList = []
        accountedSplitIdList = []
        splitList = []


        for obj in range(len(statsfullsize2)):
            idlist = []
            for i in statsfullsize2.coords[obj]:
                idlist.append(newArray[(i[0], i[1], i[2])])
            u, c = np.unique(idlist, return_counts=True)
            mode_i = statistics.mode(idlist)
            if mode_i not in accountedSplitIdList: # if the mode is not in accountedsplitlist, add to it
                accountedSplitIdList.append(mode_i)
            else:  # if it is in accountedsplitlist
                idThatSplitList.append(mode_i)
                for j in statsfullsize2.coords[obj]:
                    newArray[(j[0], j[1], j[2])] = maxx + 1
                splitIdList.append(maxx+1)
                splitList.append([mode_i, maxx + 1, time + 1])
                # print(f'{mode_i} splitted into {mode_i} and {maxx + 1} at t={time + 1}')
                maxx += 1

        # split_id_dict[time+1] = sorted([int(identity) for identity in splitIdList])
        # id_that_split_dict[time+1] = sorted([int(identity) for identity in idThatSplitList])

        idlist_now = []
        deathList = []
        for obj2 in range(len(statsfullsize2)):
            idlist_now.append(newArray[(statsfullsize2.coords[obj2][0][0], statsfullsize2.coords[obj2][0][1],
                                        statsfullsize2.coords[obj2][0][2])])
        all_id_dict[time+1] = sorted([int(identity) for identity in idlist_now])

        truemerged_id_list = [i_d for i_d in merged_id_list if i_d not in all_id_dict[time+1]]
        splitmerge_id_list = [i_d for i_d in merged_id_list if i_d in all_id_dict[time+1]]

        merged_id_dict[time + 1] = sorted([int(identity) for identity in merged_id_list])
        truemerged_id_dict[time + 1] = sorted([int(identity) for identity in truemerged_id_list])
        splitmerge_id_dict[time + 1] = sorted([int(identity) for identity in splitmerge_id_list])


        newTotal = len(idlist_now)
        oldTotal = len(idlist_previous)

        trackedList = []
        newList = []
        for theId in idlist_now:
            if theId in idlist_previous:
                trackedList.append([theId, time + 1])
            else:
                newList.append([theId, time + 1])
        # moved below after filtering (s and m filters)
        # deadIdList=[]
        # for oldId in idlist_previous:
        #     if oldId not in idlist_now and oldId not in merged_id_dict[time+1]:
        #         deadIdList.append(oldId)
        #         deathList.append([oldId, time + 1])
        # dead_id_dict[time+1]=sorted([int(identity) for identity in deadIdList])

        # events_counts[time - startpoint + 1, :] = [len(all_id_dict[time+1]), len(split_id_dict[time + 1]),
        #                                            len(merged_id_dict[time + 1]), len(birth_id_dict[time + 1]),
        #                                            len(dead_id_dict[time + 1]), len(truemerged_id_dict[time+1]), len(splitmerge_id_dict[time+1])]

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        finalv, finalc = np.unique(newArray[newArray > 0], return_counts=True)
        vcdict = {v: c for v, c in list(zip(finalv, finalc))}
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # THis block of code ensures the bigger object retains the ID during split.
        # finalv, finalc = np.unique(newArray[newArray > 0], return_counts=True)
        # vcdict = {v: c for v, c in list(zip(finalv, finalc))}

        for sfrom, soff, _ in splitList:  # If the bigger object is treated by the algorithm as secondary --> swap
            if vcdict[soff] > vcdict[sfrom]:
                indexfrom = np.where(newArray == sfrom)
                indexoff = np.where(newArray == soff)

                newArray[indexfrom] = soff
                newArray[indexoff] = sfrom

                for sublist in mergedList:
                    if sublist[0]==sfrom:
                        sublist[0]=soff
                    elif sublist[0]==soff:
                        sublist[0]=sfrom
                    if sublist[1] == sfrom:
                        sublist[1] = soff
                    elif sublist[1] == soff:
                        sublist[1] = sfrom

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$SPLIT FILTER$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        splitIdList = []
        idThatSplitList = []

        # THis block of code applies this filter to split events: pixel count of products < 1.0 * pixel count of orig
        finalv, finalc = np.unique(newArray[newArray > 0], return_counts=True)
        vcdict = {v: c for v, c in list(zip(finalv, finalc))}

        try:
            previous_t = niftireadu32(folder + t1 + '/Fullsize_2_aftertracking_' + t1 + '.nii')
        except FileNotFoundError:
            previous_t = niftireadu32(addr1 + 'Fullsize_label_' + t1 + '.nii')

        previous_v, previous_c = np.unique(previous_t[previous_t > 0], return_counts=True)
        previous_vcdict = {v: c for v, c in list(zip(previous_v, previous_c))}


        removeSplitList = []
        splitBeforeAfterSize = []

        for [sfrom, soff, _] in splitList:
            splitIdList.append(soff)
            idThatSplitList.append(sfrom)

            previous_sfrom_size = previous_vcdict[sfrom]
            current_total_size = vcdict[sfrom] + vcdict[soff]

            if 1.3*previous_sfrom_size < current_total_size:
                splitIdList.pop()
                idThatSplitList.pop()
                splitBeforeAfterSize.append([sfrom, soff, _] + [int(previous_sfrom_size), int(current_total_size), 'removed'] )

                removeSplitList.append([sfrom, soff, _])
                bornlist.append(soff)
                globalTargetIdList.append(soff)
            else:
                splitBeforeAfterSize.append([sfrom, soff, _] + [int(previous_sfrom_size), int(current_total_size), 'kept'])

        globalSplitFilterList += splitBeforeAfterSize

        for xRemove in removeSplitList:
            splitList.remove(xRemove)

        split_id_dict[time+1] = sorted([int(identity) for identity in splitIdList])
        id_that_split_dict[time+1] = sorted([int(identity) for identity in idThatSplitList])

        birth_id_dict[time + 1] = sorted([int(identity) for identity in bornlist])


        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        # We need to isolate true merge from split-and-merge. True merge means two object combine to form one object.
        # There is one less object. Split-and-merge means a fraction of one object splits and merges with another object
        # at the same time. Does not change object count.

        trueMergedList = [[mergedIntoId, mergedId,mergeTime] for (mergedIntoId, mergedId,mergeTime) in mergedList if mergedId not in idlist_now]
        splitAndMergeList = [[mergedIntoId, mergedId,mergeTime] for (mergedIntoId, mergedId,mergeTime) in mergedList if mergedId in idlist_now]

        ################MERGE FILTER##################
        removeMergeList = []
        mergeBeforeAfterSize = []

        for [pri, sec, _] in trueMergedList:
            prevvar = 0
            prevprivar = 0
            try:
                prevvar = previous_vcdict[sec]
                prevprivar = previous_vcdict[pri]
            except:
                pass
            previous_total_size = prevprivar + prevvar
            current_pri_size = vcdict[pri]

            if 0.75*(previous_total_size) > current_pri_size:
                # splitIdList.pop()
                # idThatSplitList.pop()
                mergeBeforeAfterSize.append([pri, sec, _] + [int(previous_total_size), int(current_pri_size), 'remove'] )
                removeMergeList.append([pri, sec, _])
                deathList.append(sec)
                # globalTargetIdList.append(soff)
            else:
                mergeBeforeAfterSize.append([pri, sec, _] + [int(previous_total_size), int(current_pri_size), 'keep'])

        globalMergeFilterList += mergeBeforeAfterSize

        for xRemove in removeMergeList:
            trueMergedList.remove(xRemove)
        ##############################################

        truemerge_as_primary_id_list = [int(mergedIntoId) for (mergedIntoId, mergedId,mergeTime) in trueMergedList]
        truemerge_as_secondary_id_list = [int(mergedId) for (mergedIntoId, mergedId,mergeTime) in trueMergedList]

        truemerge_as_primary_id_dict[time+1] = truemerge_as_primary_id_list
        truemerge_as_secondary_id_dict[time+1] = truemerge_as_secondary_id_list


        ### m oved from above###
        deadIdList=[]
        for oldId in idlist_previous:
            if oldId not in idlist_now and oldId not in truemerge_as_secondary_id_dict[time+1]:
                deadIdList.append(oldId)
                deathList.append([oldId, time + 1])
        dead_id_dict[time+1]=sorted([int(identity) for identity in deadIdList])

        ######

        globalSplitList.append(splitList)
        globalMergeList.append(mergedList)
        globalTrackList.append(trackedList)
        globalDeathList.append(deathList)
        globalBirthList.append(newList)
        globalIdList.append(idlist_now)

        globalTrueMergeList.append(trueMergedList)
        globalSplitAndMergeList.append(splitAndMergeList)

        events_counts[time - startpoint + 1, :] = [len(all_id_dict[time + 1]), len(split_id_dict[time + 1]),
                                                   len(merged_id_dict[time + 1]), len(birth_id_dict[time + 1]),
                                                   len(dead_id_dict[time + 1]), len(truemerged_id_dict[time + 1]),
                                                   len(splitmerge_id_dict[time + 1])]

        niftiwriteu32(newArray, folder + t2 + '/Fullsize_2_aftertracking_' + t2 + '.nii')

        max_old = maxx
    import csv
    with open(folder + 'split_filtering.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Writing the data
        csvwriter.writerow(['parent','child','time','before size','after size total', 'filtered'])
        csvwriter.writerows(globalSplitFilterList)
    with open(folder + 'merge_filtering.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Writing the data
        csvwriter.writerow(['Primary','Secondary','time','before size','after size total', 'filter'])
        csvwriter.writerows(globalMergeFilterList)


    jdead = json.dumps(dead_id_dict)
    with open(folder + 'dead_id.json', 'w') as f:
        f.write(jdead)
    jmerged = json.dumps(merged_id_dict)
    with open(folder + 'merged_id.json', 'w') as f:
        f.write(jmerged)
    jidthatsplit = json.dumps(id_that_split_dict)
    with open(folder + 'splitting_id.json', 'w') as f:
        f.write(jidthatsplit)
    jsplit = json.dumps(split_id_dict)
    with open(folder + 'split_into_id.json', 'w') as f:
        f.write(jsplit)
    jborn = json.dumps(birth_id_dict)
    with open(folder + 'born_id.json', 'w') as f:
        f.write(jborn)
    jall = json.dumps(all_id_dict)
    with open(folder + 'all_id.json', 'w') as f:
        f.write(jall)
    jtruemerge = json.dumps(truemerged_id_dict)
    with open(folder + 'all_true_merge.json', 'w') as f:
        f.write(jtruemerge)
    jtruemergeprimary = json.dumps(truemerge_as_primary_id_dict)
    with open(folder + 'truemerge_as_primary.json','w') as f:
        f.write(jtruemergeprimary)
    jtruemergesecondary = json.dumps(truemerge_as_secondary_id_dict)
    with open(folder + 'truemerge_as_secondary.json','w') as f:
        f.write(jtruemergesecondary)

    events_countsDF = pd.DataFrame(events_counts,  columns=['total','split','merged','born','dead','true_merge','split_and_merge'])
    events_countsDF.to_csv(folder + 'eventObjectCount.csv', index=False)

    maxid = np.amax(globalIdList[len(globalIdList) - 1])

    ################################################################ Combining Tracking Results into a single file ####
    starline()
    print('Combining tracking results.....')

    tempMat = niftireadu32(track_op_folder + str(startpoint) + '/' + 'Fullsize_label_' + str(startpoint) + '.nii')
    x, y, z = np.shape(tempMat)
    finalMatrix = np.zeros(shape=(x, y, z, endpoint - startpoint + 2))
    # print(np.shape(finalMatrix))
    for timepoint in range(startpoint, endpoint + 2):
        if timepoint == startpoint:
            tMatrix = niftireadu32(
                track_op_folder + str(timepoint) + '/' + 'Fullsize_label_' + str(startpoint) + '.nii')
        else:
            tMatrix = niftireadu32(
                track_op_folder + str(timepoint) + '/Fullsize_2_aftertracking_' + str(timepoint) + '.nii')

        finalMatrix[:, :, :, timepoint - startpoint] = tMatrix

    # # THis block of code ensures the bigger object retains the ID during split.
    # finalv, finalc = np.unique(newArray[newArray > 0], return_counts=True)
    # vcdict = {v: c for v, c in list(zip(finalv, finalc))}
    # for sfrom, soff, _ in splitList:  # If the bigger object is treated by the algorithm as secondary --> swap
    #     if vcdict[soff] > vcdict[sfrom]:
    #         indexfrom = np.where(newArray == sfrom)
    #         indexoff = np.where(newArray == soff)
    # 
    #         newArray[indexfrom] = soff
    #         newArray[indexoff] = sfrom

    niftiwriteu32(finalMatrix, track_op_folder + 'TrackedCombined.nii')

    globalSplitList = [[splitEvent for splitEvent in timeSplitList if splitEvent[0] != 1] for timeSplitList in globalSplitList]


# This section is constantly causing problem. Probably needs a redo
    # print(globalIdList)
    # lst = [] #      Temporarily COmmented. Uncomment once done.
    # for i in range(1, len(globalIdList) + 1):
    #     if i == 1:
    #         lst.append(str(i))
    #     else:
    #         lst.append(str(i))
    #         lst.append(str(i))
    # mydf = pd.DataFrame(nan_2d(100, len(globalIdList) * 2 - 1))
    # # print(mydf.head())
    # parent = 0
    # print(maxid)
    # # print(len(globalIdList))
    # for idd in range(2, maxid):
    #     if idd % 100 == 0:
    #         if idd % 3000 == 0:
    #             print('#', end='\n')
    #         print('#', end='')
    #     for tt in range(len(globalSplitList) + 1):
    #         if idd in globalIdList[tt]:
    #             if tt == 0:
    #                 mydf.loc[idd - 2, tt] = int(idd)
    #             else:
    #                 mydf.loc[idd - 2, tt * 2 - 1] = int(idd)
    #                 mydf.loc[idd - 2, tt * 2] = int(idd)
    #                 if idd not in globalIdList[tt - 1]:  # if the object id is new
    #                     # if idd in (np.array(globalSplitList[tt - 1])[:, 0:2].flatten()):  #replaced with next line
    #                     # if the object split from another id
    #                     # list comprehension removes the empty sublist so that [:,0:2] doesn't run into an error
    #                     # temporary__ =
    #                     try:
    #                         if idd in (np.array([sublist for sublist in globalSplitList[tt - 1] if sublist])[:, 0:2].flatten()):
    #                             for splitEvent in globalSplitList[tt - 1]:  # for all split events at that time
    #                                 if splitEvent[1] == idd:  # if idd in an event
    #                                     parent = splitEvent[0]  # get parent id
    #                                     for parent_t in range(tt):  # for all times until tt
    #                                         if tt == 1:
    #                                             if not pd.isna(mydf.loc[parent - 2, parent_t]):
    #                                                 mydf.loc[idd - 2, parent_t] = mydf.loc[parent - 2, parent_t]
    #                                         if tt > 1:
    #                                             if parent_t == 0 and not pd.isna(mydf.loc[parent - 2, parent_t]) and not \
    #                                             mydf.iloc[parent - 2, parent_t] == 'new':
    #                                                 mydf.loc[idd - 2, parent_t] = mydf.loc[parent - 2, parent_t]
    #                                             if parent_t > 0 and not pd.isna(
    #                                                     mydf.loc[parent - 2, parent_t * 2 - 1]) and not mydf.iloc[
    #                                                                                                         parent - 2, parent_t * 2 - 1] == 'new':
    #                                                 mydf.loc[idd - 2, parent_t * 2 - 1] = mydf.loc[
    #                                                     parent - 2, parent_t * 2 - 1]
    #                                             if parent_t > 0 and not pd.isna(mydf.loc[parent - 2, parent_t * 2]) and not \
    #                                                     mydf.iloc[parent - 2, parent_t * 2] == 'new':
    #                                                 mydf.loc[idd - 2, parent_t * 2] = mydf.loc[parent - 2, parent_t * 2]
    #                         else:
    #                             mydf.loc[idd - 2, tt * 2 - 2] = 'new'
    #                     except:
    #                         None
    #
    # # mydf.columns = lst  # NEED to fix this
    #
    # mydf.to_csv(folder + 'tracking_result.csv', index=False)


    flatTrueMergeList = [item for sublist in globalTrueMergeList for item in sublist]
    trueMergeDF = pd.DataFrame(flatTrueMergeList)

    flatSplitAndMergeList = [item for sublist in globalSplitAndMergeList for item in sublist]
    splitAndMergeDF = pd.DataFrame(flatSplitAndMergeList)

    allMergeList = [item for sublist in globalMergeList for item in sublist]
    allMergeDF = pd.DataFrame(allMergeList)

    ###############################################################################
    mergeCols = ['Merged Into', 'Merged', 'Time']
    if trueMergeDF.empty:
        trueMergeDF = pd.DataFrame(columns=['Merged Into','Merged','Time'])
    else:
        trueMergeDF.columns = mergeCols

    if splitAndMergeDF.empty:
        splitAndMergeDF = pd.DataFrame(columns=['Merged Into','Merged','Time'])
    else:
        splitAndMergeDF.columns = mergeCols

    if allMergeDF.empty:
        allMergeDF = pd.DataFrame(columns=['Merged Into','Merged','Time'])
    else:
        allMergeDF.columns = mergeCols
    ###############################################################################


    trueMergeDF.to_csv(folder + 'merge_list.csv', index=False)
    splitAndMergeDF.to_csv(folder + 'split-and-merge_list.csv', index=False)
    allMergeDF.to_csv(folder + 'ture-merge_+_split-and-merge_list.csv', index=False)

    flatSplitList = [item for sublist in globalSplitList for item in sublist]
    splitDF = pd.DataFrame(flatSplitList)
    if splitDF.empty:
        splitDF = pd.DataFrame(columns=['Splitted','Splitted Into','Time'])
    else:
        splitCols = ['Splitted', 'Splitted Into', 'Time']
        splitDF.columns = splitCols
    splitDF.to_csv(folder + 'split_list.csv', index=False)

    ################################################################

    # DataFrameM titles
    lst = []
    for i in range(1, len(globalTrueMergeList) + 2):
        if i == 1 or i == len(globalTrueMergeList) + 1:
            lst.append(str(i))
        else:
            lst.append(str(i))
            lst.append(str(i))

    # DataFrameS&M titles
    lst2 = []
    for i in range(1, len(globalTrueMergeList) + 2):
        if i == 1:
            lst2.append(str(i))
        else:
            lst2.append(str(i) + '_split')
            lst2.append(str(i) + '_merge')

    allMerges = [item for sublist in globalTrueMergeList for item in sublist]
    mrgDF = pd.DataFrame(nan_2d(maxid, len(globalTrueMergeList) * 2 - 1))
    combinedDF = pd.DataFrame(nan_2d(maxid, len(globalTrueMergeList) * 2 - 1))

    for eachMerge in allMerges:
        lion = eachMerge[0]
        deer = eachMerge[1]
        clock = eachMerge[2]
        mrgDF.loc[lion - 2, (clock - 1) * 2] = deer
        combinedDF.loc[lion - 2, (clock - 1) * 2] = deer
    mrgDF.columns = lst[:len(mrgDF.columns)] # Check this and fix if necessary
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

    sptDF.columns = lst[:len(sptDF.columns)]    # Check and fix this if needed
    print('Saving Split Events...')
    sptDF.to_csv(folder + 'Sheet11.csv', index=False)

    print('Saving Combined Events...')
    combinedDF.columns = lst2[:len(combinedDF.columns)]
    combinedDF.to_csv(folder + 'Sheet12.csv', index=False)

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
                globalIdExistForArray[anId - 1, 3] = parent[0]   # Stumbled upon a case with two parents. Needs attention.
            else:
                globalIdExistForArray[anId - 1, 3] = anId

            tx += 1

    globalIdExistForArray = np.uint16(globalIdExistForArray)
    # print(globalIdExistForArray[320:370])

    globalIdExistForDF = pd.DataFrame(globalIdExistForArray)
    globalIdExistForDF.columns = ['index', 'timestart', 'timeend', 'parent']

    print('Saving when the objects exist...')
    globalIdExistForDF.to_csv(folder + 'Sheet3.csv', index=False)

    print('Saving all possible target IDs...')
    globalTargetIdListDF = pd.DataFrame(globalTargetIdList)
    globalTargetIdListDF.columns = ['targetId']
    globalTargetIdListDF.to_csv(folder + 'target_IDs.csv', index=False)

    print('Generating events and intensity plots...')
    # print(track_op_folder)
    # print(imageFolder)
    # print(imageNameO)

    spDF = pd.read_csv(seg_op_folder + 'segmentation_parameters.csv')
    sT = spDF['startTime'].loc[0] - 1
    eT = spDF['endTime'].loc[0]

    # createEventsAndIntensityPlots(segpath=seg_op_folder, filePath=track_op_folder, modelName=modelName,
    #                               originalImage=imageNameO, startpoint=startpoint, endpoint=endpoint, sT=sT, eT=eT,
    #                               sTime=stime)
    # createEventsAndIntensityPlots(filePath=track_op_folder, originalImage=imageFolder + '/' + imageNameOnly.split('.')[0]+'.tif', nameOnly=imageNameOnly.split('.')[0], distance=endpoint-startpoint)


    # runAnalysis(origImgPath=imageNameO,
    #             trackedimagepath=track_op_folder + 'TrackedCombined.nii',
    #             sT=sT, eT=sT + endpoint,
    #             plotsavepath=track_op_folder[:-1])

    # It is better to create the database first then generate all plots from there.
    # print('\n\nRunning size-dependent analysis...')
    # runAnalysisNewWay(origImgPath=imageNameO,
    #             trackedimagepath=track_op_folder + 'TrackedCombined.nii',
    #             sT=sT, eT=sT + endpoint,
    #             plotsavepath=track_op_folder[:-1])

    from create_database import runSizeIntensityAnalysis
    runSizeIntensityAnalysis(dbpath=track_op_folder[:-1],sT=sT,trackedimagepath=track_op_folder + 'TrackedCombined.nii', origImgPath=imageNameO)

    print('Tracking and analysis complete. \nGenerating 3D projections of tracked image...')

    def niftireadu16(arg):
        return np.asarray(nib.load(arg).dataobj).astype(np.uint16).squeeze()
    tI_fig = niftireadu16(track_op_folder + 'TrackedCombined.nii')
    savePath3D = track_op_folder + '3DProjection'
    if not os.path.isdir(savePath3D):
        os.makedirs(savePath3D)
    colors = ["coral", "blue", "brown", "chartreuse", "aquamarine", "cyan", "darkorange", "darkred",
              "dodgerblue", "firebrick", "forestgreen", "fuchsia", "gold", "green", "hotpink", "indigo",
              "lime", "magenta", "maroon", "mediumblue", "mediumspringgreen", "navy", "olive", "orange",
              "orangered",
              "orchid", "peru", "purple", "red", "royalblue", "saddlebrown", "seagreen", "sienna", "skyblue",
              "springgreen", "teal", "tomato", "turquoise", "violet", "yellow", "yellowgreen",
              "burlywood", "cadetblue",
              "cornflowerblue",
              "darkcyan", "darkgoldenrod", "darkgreen", "darkkhaki", "darkmagenta", "darkolivegreen",
              "darkorchid", "darksalmon", "darkseagreen", "darkslateblue", "darkslategray", "darkturquoise",
              "darkviolet", "deepskyblue"
              ]

    for ii in range(tI_fig.shape[-1]):
        print(f'\r t = {ii+1}                  ', end='')
        start = datetime.now()
        fig = plt.figure(num=1, clear=True, figsize=(10, 10), constrained_layout=True)
        fig.patch.set_facecolor('white')
        ax = plt.subplot(projection='3d')
        tI_ii = tI_fig[:, :, :, ii]
        allidsinthistimepoint = np.unique(tI_ii)[1:]
        leng = len(allidsinthistimepoint)
        for j, id in enumerate(allidsinthistimepoint):
            x, y, z = np.where(tI_ii == id)
            if (len(x) > 0 and len(y) > 0):
                try:
                    label = str(id)
                    ax.plot_trisurf(x, y, z, color=colors[id % len(colors)], alpha=0.6, label=label)
                    ax.set_box_aspect([1, 1, 0.1])
                    ax.set_xlabel('x');
                    ax.set_ylabel('y');
                    ax.set_zlabel('z')
                    ax.text(np.mean(x), np.mean(y), np.mean(z), label, fontsize=6)
                except(Exception):
                    None
                ax.set_title('t=' + str(ii + 1), fontsize=20)
                ax.view_init(220, 310)
        ext = '' if ii >= 1000 else '0' if ii >= 100 else '00' if ii >= 10 else '000'
        plt.savefig(os.path.join(savePath3D, 't_' + ext + str(ii+1) + '_3D.png'), facecolor='white')
        plt.close()

    print('\n\n ##################### TRACKING COMPLETE #####################\n\n')