import numpy as np
import nibabel as nib
import pandas as pd
from functions import niftiread,niftiwriteF


def keep_values(matrix,idlist):
    newMatrix = np.copy(matrix)
    newMatrix[~np.isin(newMatrix,idlist)] = 0

    niftiwriteF(newMatrix,'/home/nirvan/Desktop/deletes/newM.nii')
    return newMatrix

def getExtremeIndices(arr):
    nonZeroIndices = np.nonzero(arr)
    n_dim = arr.ndim
    dims=[]
    for i in range(0,n_dim):
        dims.append((np.min(nonZeroIndices[i]), np.max(nonZeroIndices[i])))
    print(dims)
    return dims

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_matrix(matrix, idlist,lims, colors, ax):
    for i, id in enumerate(idlist):
        x, y, z = np.where(matrix == id)
        if(len(x)>2 and len(y)>2):
            print(id)
            ax.plot_trisurf(x, y, z, color=colors[i], alpha=0.5)
    ax.set_xlim(99,151)
    ax.set_xlim(147,180)
    ax.set_xlim(1,13)




parentId = 57
df = pd.read_csv('/home/nirvan/Desktop/AAAA/bazF/csvFiles/ft_data_tid_57.csv')
# df = df[abs(df.iloc[:, 2] - df.iloc[:, 3]) >= 4]
idlist = df.iloc[:,1].tolist()

matrix = niftiread('/home/nirvan/Desktop/AAAA/bazT/TrackedBaz.nii')
newMatrix = keep_values(matrix, idlist)

indicesRange = getExtremeIndices(newMatrix)

colors = ['red', 'blue', 'green', 'yellow','red', 'blue', 'green', 'yellow','red', 'blue', 'green', 'yellow','red', 'blue', 'green', 'yellow','red', 'blue', 'green', 'yellow','red', 'blue', 'green', 'yellow']
# fig = plt.figure(figsize=(160, 90))
fig = plt.figure()
for i in range(41):
    matrix = newMatrix[:,:,:,i]
    # fig = plt.figure(i)
    ax = fig.add_subplot(6, 7, i + 1, projection='3d')
    # plot_3d_matrix(matrix, idlist, indicesRange, colors, ax)

    for j, id in enumerate(idlist):
        x, y, z = np.where(matrix == id)
        if(len(x)>2 and len(y)>2):
            # print(id)
            ax.plot_trisurf(x, y, z, color=colors[j], alpha=0.5)
    ax.set_xlim(99,151)
    ax.set_ylim(147,180)
    ax.set_zlim(1,13)

plt.show()
