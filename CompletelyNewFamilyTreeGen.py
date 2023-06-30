import pandas as pd
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from functions import niftiwriteF, niftiread
import matplotlib.colors as mc


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
    # print(dims)
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
    trackFolder = os.path.dirname(excelFile)

    saveFolder = ftFolder
    csvFolder = saveFolder+'/csvFiles'
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
    trackFolder = os.path.dirname(excelFile)
    if not os.path.exists(csvFolder):
        os.makedirs(csvFolder)

    # new code
    existForDF = pd.read_csv(trackFolder + '/Sheet3.csv')
    existForList = existForDF.to_numpy()

    targetIdsdf = pd.read_csv(trackFolder+'/target_IDs.csv')
    targetIds = list(targetIdsdf.values.flatten())

    ftlst = []

    print('\n=================================================\n')

    for tid in targetIds:
        temp1=[]
        familyMembers=[]
        for ix in range(len(existForList)):
            if existForList[ix,0]==tid or existForList[ix,3]==tid or existForList[ix,3] in temp1:
                temp1.append(existForList[ix,0])
                familyMembers.append(existForList[ix])

        transposedFamilyMembers = np.transpose(np.array(familyMembers))
        ftlst.append(transposedFamilyMembers.tolist())

        famMem = pd.DataFrame(np.array(familyMembers))
        famMem.columns = ['index','timestart','timeend','parent']
        famMem.to_csv(csvFolder + '/ft_data_tid_' + str(tid) + '.csv')

    print(np.shape(ftlst))
 # #######################################################################################################################

    # print(ftlst)
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
        # print(notplottedlist)

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
        parentId = ft[0][0]
        df = pd.read_csv(ftFolder+'/csvFiles/ft_data_tid_'+str(parentId) + '.csv')
        # df = df[abs(df.iloc[:, 2] - df.iloc[:, 3]) >= 4]


        idlist = df.iloc[:, 1].tolist()
        print(idlist)

        matrix = niftiread(trackFolder+'/TrackedCombined.nii')
        newMatrix = keep_values(matrix, idlist)

        indicesRange = getExtremeIndices(newMatrix)

        colors = ['#0A00CD', '#EE3B3B', '#8EE5EE', '#FF6103', '#458B00', '#FFB90F', '#006400', '#B23AEE', '#00BFFF',
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

        colors = [mc.hex2color(hex_code) for hex_code in colors]

        # fig = plt.figure(figsize=(160, 90))
        fig = plt.figure(num=1, clear=True, dpi=1080)
        fig.suptitle(str(ft[0]))
        totalTimes = np.shape(matrix)[-1]
        print('Saving 3D Family Tree for ID: ')
        for i in range(totalTimes):
            matrix = newMatrix[:, :, :, i]
            # fig = plt.figure(i)
            axtitle = []
            ax = fig.add_subplot(int(np.ceil(math.sqrt(totalTimes))), int(np.ceil(totalTimes/np.ceil(math.sqrt(totalTimes)))), i + 1, projection='3d')
            # plot_3d_matrix(matrix, idlist, indicesRange, colors, ax)

            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            ax.zaxis.set_tick_params(labelright=False)


            for a in range(0, len(ft[0])):
                if ft[1][a] <= i+1 and i+1 <= ft[2][a]:
                    axtitle.append(str(ft[0][a]))

            axtitle.append('t='+str(i+1))
            ax.set_title(axtitle)

            for j, id in enumerate(idlist):
                labels = [str(id) for id in idlist]
                x, y, z = np.where(matrix == id)
                if (len(x) > 2 and len(y) > 2):
                    try:
                        label = str(id)
                        ax.plot_trisurf(x, y, z, color=colors[j], alpha=0.6, label=label)

                        # ax.text(x=int((indicesRange[0][1] - indicesRange[0][0]) / 2), y=int(indicesRange[1][1] - 10-j*10), z=indicesRange[2][1]-1,  s=str(id))
                    except(RuntimeError):
                        None

            ax.set_xlim(indicesRange[0][0]-10, indicesRange[0][1]+10)
            ax.set_ylim(indicesRange[1][0]-10, indicesRange[1][1]+10)
            ax.set_zlim(indicesRange[2][0]-2, indicesRange[2][1]+2)



        prefix = '00' if parentId < 10 else '0' if parentId < 100 else ''
        filename = ftFolder + '/FamilyTrees_3D/' + 'FT3D_ID_' + prefix + str(parentId) + '_' + str(ft[0]) + '.png'
        plt.savefig(filename)

        print(str(ft[0]), end='\r')

    print('Done!!!')

