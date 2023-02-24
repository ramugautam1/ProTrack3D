import pandas as pd
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os


def intersect(a, b):
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    aux = np.concatenate((a1, b1))
    aux.sort()
    c = aux[:-1][aux[1:] == aux[:-1]]
    return c

targetID = 14

saveFolder = '/home/nirvan/Desktop/Projects/EcadMyo_08_all/Tracking_Result_EcadMyo_08/FamilyTrees'
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)

df = pd.read_excel(
    '/home/nirvan/Desktop/Projects/EcadMyo_08_all/Tracking_Result_EcadMyo_08/TrackingID2022-08-02 18:11:23.905684.xlsx',
    sheet_name='Sheet1')
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
keys = np.asarray(x.keys())[1:].astype(np.int32)
vals = np.asarray(x.values)[1:].astype(np.int32)

for i, v in enumerate(vals):
    if v > 10:
        lst2.append(keys[i])

lst2.sort()

targetIds = intersect(lst, lst2)
print('\n')
# print(np.size(targetIds), 'family trees.')
# print(list(targetIds))
ftlst = []

ftnum = 0

print('\n=================================================\n')

for tid in [targetID]:

    set = []
    indexlist = []
    print("target id: " + str(tid), end=' ')
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

    print(u'\u2713') if len(indexlist) == len(timelist) == len(timeendlist) == len(parentlist) else print('error')
    set.append(indexlist)
    set.append(timelist)
    set.append(timeendlist)
    set.append(parentlist)

    ftlst.append(set)

print('\n=================================================\n')
# for a in range(0,len(ftlst)):
# 	for b in range(0,len(ftlst[a])):
# 		print(ftlst[a][b])
# 	print('')

# print('\n=================================================\n')
print(ftlst)
print('\n=================================================\n')

colors = ['#0000CD', '#EE3B3B', '#8EE5EE', '#FF6103', '#458B00', '#FFB90F', '#006400', '#B23AEE', '#00BFFF', '#00C957',
          '#8B6914',
          '#FF1493', '#8FBC8F', '#CD661D', '#8B8878', '#FF7256', '#0000CD', '#EE3B3B', '#8EE5EE', '#FF6103', '#458B00',
          '#FFB90F',
          '#E06E00', '#B23EEE', '#E0BFFF', '#0EC957', '#8E6914', '#FA1493', '#EFBC8F', '#CE661D', '#8E8878', '#FE7256',
          '#EE3B3B',
          '#8EE5EE', '#FF6103', '#458B00', '#FFB90F', '#006400', '#B23AEE', '#00BFFF', '#00C957', '#8B6914', '#FF1493',
          '#8FBC8F',
          '#CD661D', '#8B8878', '#FF7256', '#0000CD', '#EE3B3B', '#8EE5EE', '#FF6103', '#458B00', '#FFB90F', '#E06E00',
          '#B23EEE',
          '#E0BFFF', '#0EC957', '#8E6914', '#FA1493', '#EFBC8F', '#CE661D', '#8E8878', '#FE7256', '#8E6914', '#FA1493',
          '#EFBC8F']

for index, ft in enumerate(ftlst):
    fig = plt.figure(figsize=(52, 27))
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
        if ind != 0 and (itm not in ft[3]) and ((ft[2][ind] - ft[1][ind]) < 4):
            notplottedlist.append(ind)
    print(notplottedlist)

    for i, j in enumerate((ft[0])):
        if i not in notplottedlist:  # min time filter
            for k in range(ft[1][i] - 1, ft[2][i]):
                ax.scatter(k + 1, i + 1, c=colors[i], s=400)
                ax.text(ft[2][i] + 1, i + 1, str(ft[0][i]), fontsize=30 if i == 0 else 20)
            plt.plot()
            for iii in range(1, len(ft[0])):
                if iii not in notplottedlist:  # min time filter
                    l = ft[0].index(ft[3][iii])
                    plt.plot([ft[1][iii] - 1, ft[1][iii]], [l + 1, iii + 1], c=colors[iii], linewidth=1)
            plt.plot([ft[1][i], ft[2][i]], [i + 1, i + 1], c='k', linewidth=1)

    prefix = '00' if ft[0][0] < 10 else '0' if ft[0][0] < 100 else ''
    filename = saveFolder + '/' + 'FT_ID_' + prefix + str(ft[0][0]) + '.png'
    plt.savefig(filename)
    plt.close(fig)



