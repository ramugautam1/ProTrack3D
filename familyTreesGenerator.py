import pandas as pd
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from functions import niftiwriteF, niftiread


def keep_values(matrix, idlist):
    newMatrix = np.copy(matrix)
    newMatrix[~np.isin(newMatrix, idlist)] = 0

    # niftiwriteF(newMatrix, '/home/nirvan/Desktop/deletes/newM.nii')
    return newMatrix


def getExtremeIndices(arr):
    nonZeroIndices = np.nonzero(arr)
    n_dim = arr.ndim
    dims = []
    for i in range(0, n_dim):
        dims.append((np.min(nonZeroIndices[i]), np.max(nonZeroIndices[i])))
    print(dims)
    return dims


def plot_3d_matrix(matrix, idlist, lims, colors, ax):
    for i, id in enumerate(idlist):
        x, y, z = np.where(matrix == id)
        if (len(x) > 2 and len(y) > 2):
            print(id)
            ax.plot_trisurf(x, y, z, color=colors[i], alpha=0.5)
    ax.set_xlim(99, 151)
    ax.set_xlim(147, 180)
    ax.set_xlim(1, 13)

def intersect(a, b):
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    aux = np.concatenate((a1, b1))
    aux.sort()
    c = aux[:-1][aux[1:] == aux[:-1]]
    return c


def generateFamilyTrees(excelFile, ftFolder):
    # saveFolder = '/home/nirvan/Desktop/Projects/EcadMyo_08_all/Tracking_Result_EcadMyo_08/FamilyTrees'
    saveFolder = ftFolder
    csvFolder = saveFolder+'/csvFiles'
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
    trackFolder = os.path.dirname(excelFile)
    if not os.path.exists(csvFolder):
        os.makedirs(csvFolder)

    # #get mergelist
    # df1 = pd.read_excel(excelFile, sheet_name='Sheet10')
    # df = pd.read_excel(excelFile, sheet_name='Sheet10', usecols=[i for i in range(1, df1.shape[1], 2)])
    # df = df1.iloc[:, 1::2]
    # df2 = pd.read_excel(excelFile, sheet_name='Sheet1')
    # mergelist_all = []
    # for i, row in enumerate(df.values):
    #     for j, val in enumerate(row):
    #         if not pd.isna(val):
    #             k = 0
    #             if k == 0:
    #                 for ix, rowx in enumerate(df2.values):
    #                     for jx, valx in enumerate(rowx):
    #                         if k == 0:
    #                             if not pd.isna(valx) and valx == val:
    #                                 k = 1
    #                                 thetime = int(df2.columns[jx])
    #
    #             mergelist_all.append((int(df.columns[j]), int(val), i + 2, thetime))
    # print(str(len(mergelist_all)) + ' merge events in total.')
    #
    # merge_time_list = []
    # merged_list = []
    # merged_into_list = []
    # merged_id_starttime = []
    # for event in mergelist_all:
    #     merge_time_list.append(event[0])
    #     merged_list.append(event[1])
    #     merged_into_list.append(event[2])
    #     merged_id_starttime.append(event[3])
    #
    # temp_df = pd.DataFrame(mergelist_all, columns=['merge_time', 'merged', 'merged_into', 'merged_id_starttime'])
    # # drop rows where merged and merged_into columns are identical
    # mergelist_all_df = temp_df.drop_duplicates(subset=['merged', 'merged_into']).reset_index(drop=True)
    #
    # mergelist_all_df.to_csv(csvFolder + '/mergelist.csv', index=False)
    #############################################################################################

    # df = pd.read_excel('/home/nirvan/Desktop/Projects/EcadMyo_08_all/Tracking_Result_EcadMyo_08/TrackingID2022-08-02 18:11:23.905684.xlsx', sheet_name='Sheet1')
    df = pd.read_excel(excelFile, sheet_name='Sheet1')
    # print(df.head())
    lst = []
    # print(df.iloc[0:10,0:10])

    for ix in range(0, df.shape[0]):
        if (not pd.isna(df.iloc[ix, 0]) and df.iloc[ix, 0] == ix + 2) or (df.iloc[ix, :] == 'new').any():
            lst.append(ix + 2)

    # print(lst)
    # print('\n\n')

    lst2 = []

    x = df.stack().value_counts()
    x.pop('new')
    keys = np.asarray(x.keys())[1:].astype(np.int32)
    vals = np.asarray(x.values)[1:].astype(np.int32)

    for i, v in enumerate(vals):
        if v > 10:
            lst2.append(keys[i])

    lst2.sort()

    targetIds = intersect(lst, lst2)
    print('\n')
    print(np.size(targetIds), 'family trees.')
    # print(list(targetIds))
    ftlst = []

    ftnum = 0

    print('\n=================================================\n')

    for tid in targetIds:

        set = []
        indexlist = []
        print("target id: " + str(tid), end='\r')
        for ix in range(0, df.shape[0]):
            for jx in range(0, df.shape[1], 2):
                if df.iloc[ix, jx] == tid and ix + 2 not in indexlist:
                    indexlist.append(ix + 2)
                    break
        # print("indexlist:  ",indexlist)

        df2 = df.copy()
        k2 = str(int(df.shape[1] / 2 + 1)) + '.1'
        k1 = str(int(df.shape[1] / 2 + 1))
        df2[k2] = df.loc[:, k1]
        # print(df2.head())

        timelist = []
        for inndx, idx in enumerate(indexlist):
            for ix in range(idx - 2, df2.shape[0]):
                size1 = np.size(timelist)
                for jx in range(0, df2.shape[1], 2):
                    if jx == 0 and df2.iloc[ix, jx] == idx:
                        timelist.append(1)
                        break

                    elif jx > 0 and jx < (df2.shape[1] - 1):
                        if df2.iloc[ix, jx - 1] == idx:
                            timelist.append(1 + (jx / 2))
                            break

                    elif jx == df2.shape[1] - 1 and df2.iloc[ix, df2.shape[1] - 1] == idx:
                        timelist.append(1 + jx / 2)
                        break

                # print(idx,df2.iloc[ix,jx-1],df2.iloc[ix,jx-1]==idx, df2.iloc[ix,0]==idx,1+jx/2)
                size2 = np.size(timelist)
                # print(timelist, size1, size2)
                if (size2 > size1):
                    break
        timelist = [int(tl) for tl in timelist[0:len(indexlist)]]
        # print('timelist:  ', timelist)

        timeendlist = []
        for idx in indexlist:
            for ix in range(idx - 2, df.shape[0]):
                size3 = np.size(timeendlist)
                for jx in range(df.shape[1] - 1, 0, -2):
                    if df.iloc[ix, jx] == idx or df.iloc[ix, jx - 1] == idx:
                        timeendlist.append(1 + math.ceil(jx / 2))
                        break
                size4 = np.size(timeendlist)
                if (size4 > size3):
                    break
        timelist = [int(tl) for tl in timelist]
        # print('timeendlist:  ', timeendlist)

        parentlist = []
        for index, idx in enumerate(indexlist):
            if index == 0:
                parent = idx
            else:
                parent = df.iloc[idx - 2, 2 * (timelist[index] - 2)]
            parentlist.append(parent)

        # print(u'\u2713') if len(indexlist) == len(timelist) == len(timeendlist) == len(parentlist) else print('error')
        set.append(indexlist)
        set.append(timelist)
        set.append(timeendlist)
        set.append(parentlist)
        #############################################################################

        # # TRYING TO INCORPORATE MERGE DATA INTO FT
        #
        # all_merged_index=[]
        # all_merge_time_index=[]
        #
        # for id in indexlist:
        #     if id in merged_into_list:
        #         id_index = [i for i in range(len(merged_into_list)) if merged_into_list[i]==id]
        #         merged_index = [merged_list[i] for i in id_index]
        #         merge_time_index = [merge_time_list[i] for i in id_index]


        #############################################################################

        mydf = pd.DataFrame()
        mydf['index'] = indexlist
        mydf['timestart'] = timelist
        mydf['timeend'] = timeendlist
        mydf['parent'] = parentlist

        mydf.to_csv(csvFolder+'/ft_data_tid_'+str(tid)+'.csv')

    # added a tab to line just under
        ftlst.append(set)

        if tid>10:
            print(ftlst)
            break


    print(np.shape(ftlst))

    print('\n=================================================\n')
    # for a in range(0,len(ftlst)):
    # 	for b in range(0,len(ftlst[a])):
    # 		print(ftlst[a][b])
    # 	print('')

    # print('\n=================================================\n')
    print(ftlst)
    print('\n=================================================\n')

    colors = ['#0000CD', '#EE3B3B', '#8EE5EE', '#FF6103', '#458B00', '#FFB90F', '#006400', '#B23AEE', '#00BFFF',
              '#00C957', '#8B6914',
              '#FF1493', '#8FBC8F', '#CD661D', '#8B8878', '#FF7256', '#0000CD', '#EE3B3B', '#8EE5EE', '#FF6103',
              '#458B00', '#FFB90F',
              '#E06E00', '#B23EEE', '#E0BFFF', '#0EC957', '#8E6914', '#FA1493', '#EFBC8F', '#CE661D', '#8E8878',
              '#FE7256', '#EE3B3B',
              '#8EE5EE', '#FF6103', '#458B00', '#FFB90F', '#006400', '#B23AEE', '#00BFFF', '#00C957', '#8B6914',
              '#FF1493', '#8FBC8F',
              '#CD661D', '#8B8878', '#FF7256', '#0000CD', '#EE3B3B', '#8EE5EE', '#FF6103', '#458B00', '#FFB90F',
              '#E06E00', '#B23EEE',
              '#E0BFFF', '#0EC957', '#8E6914', '#FA1493', '#EFBC8F', '#CE661D', '#8E8878', '#FE7256', '#8E6914',
              '#FA1493', '#EFBC8F']

    for index, ft in enumerate(ftlst):
        fig = plt.figure(num=1, clear=True,figsize=(52, 27))
        ax = plt.subplot()
        ax.set_xlim(0, max(ft[2]) + 5)
        ax.set_ylim(0, len(ft[0]) + 1)
        ax.set_xlabel('Time Points (t)', fontsize=17, color='k')
        mpl.rc('xtick', labelsize=17)
        mpl.rc('ytick', labelsize=17)
        plt.xticks(rotation=0)
        plt.yticks(color='w')
        k = 1

        notplottedlist = []
        for ind, itm in enumerate(ft[0]):
            if ind != 0 and (itm not in ft[3]) and ((ft[2][ind] - ft[1][ind]) < 3):
                notplottedlist.append(ind)
        print(notplottedlist)

        for i, j in enumerate((ft[0])):
            if i not in notplottedlist:  # min time filter
                for k in range(ft[1][i] - 1, ft[2][i]):
                    ax.scatter(k + 1, i + 1, c=colors[i], s=400)
                    ax.text(ft[2][i] + 1, i + 1, str(ft[0][i]),
                            fontsize=30 if i == 0 else 20)
                plt.plot()
                for iii in range(1, len(ft[0])):
                    if iii not in notplottedlist:  # min time filter
                        l = ft[0].index(ft[3][iii])
                        plt.plot([ft[1][iii] - 1, ft[1][iii]],
                                 [l + 1, iii + 1], c=colors[iii], linewidth=1)
                plt.plot([ft[1][i], ft[2][i]], [
                         i + 1, i + 1], c='k', linewidth=1)

        prefix = '00' if ft[0][0] < 10 else '0' if ft[0][0] < 100 else ''
        filename = saveFolder + '/' + 'FT_ID_' + \
            prefix + str(ft[0][0]) + '.png'
        plt.savefig(filename)

########################################################################################################################
    if not os.path.isdir(ftFolder+'/FamilyTrees_3D'):
        os.makedirs(ftFolder+'/FamilyTrees_3D')

    print('Generating 3D Family Trees.')

    for ft in ftlst:
        # parentId = 57

        # df = pd.read_csv('/home/nirvan/Desktop/AAAA/bazF/csvFiles/ft_data_tid_57.csv')

        parentId = ft[0][0]
        df = pd.read_csv(ftFolder+'/csvFiles/ft_data_tid_'+str(parentId) + '.csv')
        # df = df[abs(df.iloc[:, 2] - df.iloc[:, 3]) >= 4]


        idlist = df.iloc[:, 1].tolist()

        matrix = niftiread(trackFolder+'/TrackedCombined.nii')
        newMatrix = keep_values(matrix, idlist)

        indicesRange = getExtremeIndices(newMatrix)

        colors = ['#0000CD', '#EE3B3B', '#8EE5EE', '#FF6103', '#458B00', '#FFB90F', '#006400', '#B23AEE', '#00BFFF',
                  '#00C957', '#8B6914',
                  '#FF1493', '#8FBC8F', '#CD661D', '#8B8878', '#FF7256', '#0000CD', '#EE3B3B', '#8EE5EE', '#FF6103',
                  '#458B00', '#FFB90F',
                  '#E06E00', '#B23EEE', '#E0BFFF', '#0EC957', '#8E6914', '#FA1493', '#EFBC8F', '#CE661D', '#8E8878',
                  '#FE7256', '#EE3B3B',
                  '#8EE5EE', '#FF6103', '#458B00', '#FFB90F', '#006400', '#B23AEE', '#00BFFF', '#00C957', '#8B6914',
                  '#FF1493', '#8FBC8F',
                  '#CD661D', '#8B8878', '#FF7256', '#0000CD', '#EE3B3B', '#8EE5EE', '#FF6103', '#458B00', '#FFB90F',
                  '#E06E00', '#B23EEE',
                  '#E0BFFF', '#0EC957', '#8E6914', '#FA1493', '#EFBC8F', '#CE661D', '#8E8878', '#FE7256', '#8E6914',
                  '#FA1493', '#EFBC8F']
        # fig = plt.figure(figsize=(160, 90))
        fig = plt.figure(num=1, clear=True)
        totalTimes = np.shape(matrix)[-1]
        print('Saving 3D Family Tree for ID: ')
        for i in range(totalTimes):
            matrix = newMatrix[:, :, :, i]
            # fig = plt.figure(i)
            ax = fig.add_subplot(int(np.ceil(math.sqrt(totalTimes))), int(np.ceil(totalTimes/np.ceil(math.sqrt(totalTimes)))), i + 1, projection='3d')
            # plot_3d_matrix(matrix, idlist, indicesRange, colors, ax)

            for j, id in enumerate(idlist):
                labels = [str(id) for id in idlist]
                x, y, z = np.where(matrix == id)
                if (len(x) > 2 and len(y) > 2):
                    # print(id)
                    label = str(id)
                    print('label ' + label)
                    ax.plot_trisurf(x, y, z, color=colors[j], alpha=0.5, label=label)
            ax.set_xlim(indicesRange[0][0]-10, indicesRange[0][1]+10)
            ax.set_ylim(indicesRange[1][0]-10, indicesRange[1][1]+10)
            ax.set_zlim(indicesRange[2][0]-2, indicesRange[2][1]+2)

            ax.legend(labels=labels)

        prefix = '00' if parentId < 10 else '0' if parentId < 100 else ''
        filename = ftFolder + '/FamilyTrees_3D/' + 'FT3D_ID_' + prefix + str(parentId) + '_' + str(ft[0]) + '.png'
        plt.savefig(filename)

        print(str(ft[0]), end='\r')

    print('Done!!!')

