import pandas as pd
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mc
import nibabel as nib
import json

def niftiread(arg):
    return np.asarray(nib.load(arg).dataobj).astype(np.float32).squeeze()

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
        dims.append((int(np.min(nonZeroIndices[i])), int(np.max(nonZeroIndices[i]))))
    # print(dims)
    return dims


def plot_3d_matrix(matrix, idlist, lims, colors, ax):
    for i, id in enumerate(idlist):
        x, y, z = np.where(matrix == id)
        if (len(x) > 2 and len(y) > 2):
            # print(id)
            ax.plot_trisurf(x, y, z, color=colors[i%len(colors)], alpha=0.5)
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

    trackFolder = os.path.dirname(excelFile)

    saveFolder = ftFolder
    csvFolder = saveFolder + '/csvFiles'
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
    ## For GT
    # targetIds =  [6,11,13,15,17,22,25,30,35,38,43,45,79,102,112,122,126,144,147, 169,175, 179, 184, 205, 199, 211, 220, 332, 341, 344, 351, 408, 419, 519, 522, 522, 529, 530, 537, 560, 561, 619, 620, 634, 635, 636, 637, 638, 639]


    merge_list = pd.read_csv(trackFolder + '/merge_list.csv')

    split_list = pd.read_csv(trackFolder + '/split_list.csv')

    split_list = split_list[~split_list['Splitted Into'].duplicated(keep='first')]  # Encountered one issue where two different split events resulted into same ID being generated at different timepoints. Look into this.

    # alltmin = []
    # alltmax = []
    print('\n=================================================\n')
    if not os.path.isdir(ftFolder + '/FamilyTrees_3D'):
        os.makedirs(ftFolder + '/FamilyTrees_3D')
    if not os.path.isdir(ftFolder + '/TreeDiagrams'):
        os.makedirs(ftFolder + '/TreeDiagrams')

    indices_ = {}
    matrix = niftiread(trackFolder + '/TrackedCombined.nii')
    for tid in targetIds:

        prefix = '0000' if tid < 10 else '000' if tid < 100 else '00' if tid < 1000 else '0' if tid < 10000 else ''
        print(f'\rProcessing ID {prefix}{tid}', end='', flush=True)

        temp1 = []
        familyMembers = []

        for ix in range(len(existForList)):
            if existForList[ix, 0] == tid or existForList[ix, 3] == tid or existForList[ix, 3] in temp1:
                temp1.append(existForList[ix, 0])
                familyMembers.append(existForList[ix])

        transposedFamilyMembers = np.transpose(np.array(familyMembers))

        # # ftlst.append(transposedFamilyMembers.tolist())
        # alltmin.append(min(transposedFamilyMembers[1]))
        # alltmax.append(max(transposedFamilyMembers[2]))
        famMem = pd.DataFrame(np.array(familyMembers))
        famMem.columns = ['index', 'timestart', 'timeend', 'parent']
        famMem.to_csv(csvFolder + '/split_data_tid_' + str(tid) + '.csv')

        all_ids_in_ft_s = np.array(famMem.loc[:, 'index'])

        mrglst_for_ft = merge_list[merge_list['Merged Into'].isin(all_ids_in_ft_s)]['Merged'].tolist()
        merge_t_for_ft = merge_list[merge_list['Merged Into'].isin(all_ids_in_ft_s)]['Time'].tolist()

        bigmrglst_for_ft = merge_list[merge_list['Merged'].isin(all_ids_in_ft_s)]['Merged Into'].tolist()
        bigmrg_t_for_ft = merge_list[merge_list['Merged'].isin(all_ids_in_ft_s)]['Time'].tolist()

        fam_m = []
        for i_, m_ in enumerate(mrglst_for_ft):
            fam_m.append(existForList[m_ - 1].tolist())
        # mrglst.append(mrglst_for_ft)

        splitlst_for_ft = split_list[(split_list['Splitted'].isin(all_ids_in_ft_s) ) |(split_list['Splitted Into'].isin(all_ids_in_ft_s))]

        #         bigmrglst_for_ft = merge_list[merge_list['Merged Into'].isin(all_ids_in_ft_s)]
        #         smlmrglst_for_ft = merge_list[merge_list['Merged'].isin(all_ids_in_ft_s)]
        #         allmrglst_for_ft = pd.concat([bigmrglst_for_ft, smlmrglst_for_ft], ignore_index=True).drop_duplicates()

        allmrglst_for_ft = merge_list[(merge_list['Merged Into'].isin(all_ids_in_ft_s)) | (merge_list['Merged'].isin(all_ids_in_ft_s))]

        allFamilyMembersIncludingSplitAndMerge = pd.concat \
            ([allmrglst_for_ft['Merged Into'] ,allmrglst_for_ft['Merged'] ,famMem['index'], famMem['parent']]).unique()    # added famMem['parent']

        allFamMemExistTimes = existForDF.loc[existForDF['index'].isin(allFamilyMembersIncludingSplitAndMerge)].reset_index(drop=True)

        figsize1 = (allFamMemExistTimes.timeend.max() + 2, len(allFamMemExistTimes) + 2)
        fig = plt.figure(num=1, clear=True, figsize=figsize1)
        fig.patch.set_facecolor('white')

        ax = plt.subplot()
        ax.set_ylim(0, len(allFamMemExistTimes) + 2)
        ax.set_xlim(allFamMemExistTimes.timestart.min() - 1, allFamMemExistTimes.timeend.max() + 2)
        ax.set_xlabel('timepoints (t)', fontsize=20, color='k')
        ax.set_xticks(np.arange(allFamMemExistTimes.timestart.min(), allFamMemExistTimes.timeend.max() + 1))
        mpl.rc('xtick', labelsize=20)
        mpl.rc('ytick', labelsize=20)
        plt.yticks(color='w')
        plt.xticks(rotation=0, fontsize=15)

        for i, j in enumerate(allFamMemExistTimes['index']):
            for k in range(allFamMemExistTimes['timestart'][i], allFamMemExistTimes['timeend'][i ] +1):
                ax.scatter(k, i + 1, color=colors[j % len(colors)], s=200)
            ax.text(allFamMemExistTimes['timeend'][i ] +0.25, i+ 1, str(j), fontsize=30 if i == 0 else 20)
            plt.plot([allFamMemExistTimes['timestart'][i], allFamMemExistTimes['timeend'][i]], [i + 1, i + 1],
                     color="green")

        # print(allFamMemExistTimes)
        for i, row in splitlst_for_ft.iterrows():
            sd, si, tt = row['Splitted'], row['Splitted Into'], row['Time']
            # print(sd, si, tt)
            plt.plot([tt - 1, tt], [list(allFamMemExistTimes['index']).index(sd) + 1,
                                    list(allFamMemExistTimes['index']).index(si) + 1], color="darkblue")


        for i, row in allmrglst_for_ft.iterrows():
            mi, md, tt = row['Merged Into'], row['Merged'], row['Time']
            plt.plot([tt - 1, tt], [list(allFamMemExistTimes['index']).index(md) + 1,
                                    list(allFamMemExistTimes['index']).index(mi) + 1], color='red', linestyle='--')

        prefix = '0000' if tid < 10 else '000' if tid < 100 else '00' if tid < 1000 else '0' if tid < 10000 else ''
        filename = saveFolder + '/TreeDiagrams/' + 'FT_ID_' + prefix + str(tid) + '.png'
        plt.savefig(filename)
        plt.close()

        idlist=allFamilyMembersIncludingSplitAndMerge
        parentId = tid
        # matrix = niftiread(trackFolder + '/TrackedCombined.nii')
        newMatrix = keep_values(matrix, idlist)

        indicesRange = getExtremeIndices(newMatrix)

        tmax = allFamMemExistTimes['timeend'].max()
        tmin = allFamMemExistTimes['timestart'].min()

        indices_[tid] = indicesRange + list((int(tmin),int(tmax)))

        totalTimes = tmax - tmin + 3
        figsize=(20,int(np.ceil(totalTimes/5)) * 4)

        fig = plt.figure(num=1, clear=True, figsize=figsize)
        fig.patch.set_facecolor('white')
        fig.suptitle(str(tid))
        list_coordinates_all = []
        for i in range(max(tmin-2, 0), min(tmax+1,matrix.shape[-1])):
            mtrx = newMatrix[:, :, :, i]
            axtitle = []
            subplotloc = int(np.ceil(totalTimes / 5))
            ax = fig.add_subplot(subplotloc, 5, i - tmin + 3, projection='3d')
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            ax.zaxis.set_tick_params(labelright=False)
            ax.zaxis.set_ticklabels([])

            axtitle = list(np.unique(mtrx)[1:])
            axtitle.append('t=' + str(i + 1))
            ax.set_title(str(axtitle), fontsize=10)
            allmemberidlist = list(idlist)
            ax.set_box_aspect([1, 1, 0.25])
            # legends = {allmemberidlist[i]: colors[i % len(colors)] for i in range(len(allmemberidlist))}

            for j, id in enumerate(allmemberidlist):
                labels = [str(id) for id in allmemberidlist]
                x, y, z = np.where(mtrx == id)
                if (len(x) > 2 and len(y) > 2):
                    try:
                        if (len(np.unique(z)) == 1):
                            z = [zz + 1 if az % 2 == 0 else zz for az, zz in enumerate(z)]
                        label = str(id)
                        ax.plot_trisurf(x, y, z, color=colors[id % len(colors)], alpha=0.6, label=label)
                        # ax.set_box_aspect([1, 1, 0.25])
                        # ax.text(x=int((indicesRange[0][1] - indicesRange[0][0]) / 2), y=int(indicesRange[1][1] - 10-j*10), z=indicesRange[2][1]-1,  s=str(id))
                    except(Exception):
                        None

            ax.set_xlim(indicesRange[0][0] - 4, indicesRange[0][1] + 4)
            ax.set_ylim(indicesRange[1][0] - 4, indicesRange[1][1] + 4)
            ax.set_zlim(indicesRange[2][0] - 1, indicesRange[2][1] + 1)
            list_coordinates_all.append(
                [indicesRange[0][0] - 4, indicesRange[0][1] + 4, indicesRange[1][0] - 4, indicesRange[1][1] + 4,
                 indicesRange[2][0] - 1, indicesRange[2][1] + 1]
            )
            ax.grid('off')
            ax.view_init(40, 50)
        fig.tight_layout()
        filename = ftFolder + '/FamilyTrees_3D/' + 'FT3D_ID_' + prefix + str(parentId) + '_' + str(
            list(idlist)).replace(' ',
                                  '')

        print(f'\rProcessing ID {prefix}{tid}{figsize1}{figsize}', end='', flush=True)
        filename = filename[:250]  # due to file name length limit
        plt.savefig(filename + '.png')
        plt.close()
    jind = json.dumps(indices_)
    with open(ftFolder + '/extreme_indices.json','w') as f:
        f.write(jind)
    ########################################################################################################################
