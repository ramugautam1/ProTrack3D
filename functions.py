import numpy as np
import nibabel as nib
import pandas as pd


def dashline():
    print('---------------------------------------------------------')


def starline():
    print('*********************************************************')


def niftiread(arg):
    return np.asarray(nib.load(arg).dataobj).astype(np.float32).squeeze()

def niftireadI(arg):
    return np.asarray(nib.load(arg).dataobj).astype(np.int16).squeeze()

def niftireadu16(arg):
    return np.asarray(nib.load(arg).dataobj).astype(np.uint16).squeeze()

def niftireadu32(arg):
    return np.asarray(nib.load(arg).dataobj).astype(np.uint32).squeeze()
def niftiwrite(a, b):
    nib.save(nib.Nifti1Image(np.uint32(a),affine=np.eye(4)),b)

def niftiwriteu16(a,b):
    nib.save(nib.Nifti1Image(np.uint16(a),affine=np.eye(4)),b)

def niftiwriteu32(a, b):
    nib.save(nib.Nifti1Image(np.uint32(a), affine=np.eye(4)), b)


def niftiwriteF(a, b):
    nib.save(nib.Nifti1Image(a, affine=np.eye(4)), b)


def niftiwrite8(a, b):
    nib.save(nib.Nifti1Image(np.uint8(a),affine=np.eye(4)),b)


def line(a):
    return a*50


def rand():
    return np.random.rand()


def size3(arg):
    """
    :param arg: 3d array
    :return: [x,y,z] dimensions of 3d array
    example: [x,y,z] = size3(np.random.rand(3,3,3))
    """
    return [np.size(arg, axis=0), np.size(arg, axis=1), np.size(arg, axis=2)]


def getVoxelList(arg, orgnum):
    """
    :param arg: np array of niftii file
    :param orgnum: orgnum (no. of objects)
    :return: voxellist (a pandas dataframe voxels with one feature: VoxelList, a list of id's with all its pixels)

    """

    [fx, fy, fz] = size3(arg)

    data = {
        "VoxelList": [[[]]]
    }
    voxels = pd.DataFrame(data)
    # print(f'--------------------------{orgnum}')
    # print(type(voxels.VoxelList[0]))
    for i1 in range(0, fx):
        for i2 in range(0, fy):
            for i3 in range(0, fz):
                if arg[i1, i2, i3] != 0:
                    for l in range(1, orgnum + 1):
                        if arg[i1, i2, i3] == l:
                            print(l)
                            if voxels.size < l + 1:
                                voxels.loc[l - 1, 'VoxelList'] = np.array([[i1, i2, i3]])
                            else:
                                voxels.loc[l - 1, 'VoxelList'] = np.concatenate(
                                    (np.array(voxels.VoxelList[l - 1]), np.array([[i1, i2, i3]])), axis=0)
    return voxels

def intersect(a, b):
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    aux = np.concatenate((a1, b1))
    aux.sort()
    c = aux[:-1][aux[1:] == aux[:-1]]
    return c


def setdiff(a, b):
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    return np.asarray([i for i in a1 if i not in b1])


def isempty(a):
    return True if np.size(a, axis=0) == 0 else False


def nan_2d(a,b):
    x = np.zeros(shape=(a,b))
    for i in range(np.size(x, axis=0)):
        for j in range(np.size(x, axis=1)):
            x[i, j] = np.nan
    return x
