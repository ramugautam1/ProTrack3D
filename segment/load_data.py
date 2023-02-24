import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.ndimage
def loadDataGeneral(df, path,img_size):
    """
    This function loads data stored in nifti format. Data should already be of
    appropriate shape.

    Inputs:
    - df: Pandas dataframe with two columns: image filenames and ground truth filenames.
    - path: Path to folder containing filenames from df.
    - append_coords: Whether to append coordinate channels or not.
    Returns:
    - X: Array of 3D images with 1 or 4 channels depending on `append_coords`.
    - y: Array of 3D masks with 1 channel.
    """
    X, y = [], []
    for i, item in df.iterrows():
        img = nib.load(path + item[0]).get_data()
        mask = nib.load(path + item[1]).get_data()
        mask = np.clip(mask, 0, 255)
        out = mask
        img = img#*1./33240
        img = scipy.ndimage.zoom(img, [img_size/img.shape[0],img_size/img.shape[1], 1], order=1)
        out = scipy.ndimage.zoom(out, [img_size / out.shape[0], img_size / out.shape[1], 1], order=1)

        X.append(img)
        y.append(out)
        #cmask = (mask * 1. / 255)

    X = np.expand_dims(X, -1)
    y = np.expand_dims(y, -1)
    #y = np.concatenate((1 - y, y), -1)
    #y = np.array(y)
    print('### Dataset loaded')
    print('\t{}'.format(path))
    #print('\t{}\t{}'.format(X.shape, y.shape))
    #print('\tX:{:.1f}-{:.1f}\ty:{:.1f}-{:.1f}\n'.format(X.min(), X.max(), y.min(), y.max()))
    return X, y


def loadDataShow(df, path,img_size):
    """
    This function loads data stored in nifti format. Data should already be of
    appropriate shape.

    Inputs:
    - df: Pandas dataframe with two columns: image filenames and ground truth filenames.
    - path: Path to folder containing filenames from df.
    - append_coords: Whether to append coordinate channels or not.
    Returns:
    - X: Array of 3D images with 1 or 4 channels depending on `append_coords`.
    - y: Array of 3D masks with 1 channel.
    """
    X, y = [], []
    for i, item in df.iterrows():
        img = nib.load(path + item[0]).get_data()
        mask = nib.load(path + item[1]).get_data()
        mask = np.clip(mask, 0, 255)
        #cmask = (mask * 1. / 255)
        out = mask
        img = scipy.ndimage.zoom(img, [img_size / img.shape[0], img_size / img.shape[1], 1], order=1)
        out = scipy.ndimage.zoom(out, [img_size / out.shape[0], img_size / out.shape[1], 1], order=1)
        X.append(img)
        y.append(out)
    X = np.expand_dims(X, -1)
    y = np.expand_dims(y, -1)
    y = np.concatenate((1 - y, y), -1)
    y = np.array(y)
    print('### Dataset loaded')
    print('\t{}'.format(path))
    #print('\t{}\t{}'.format(X.shape, y.shape))
    #print('\tX:{:.1f}-{:.1f}\ty:{:.1f}-{:.1f}\n'.format(X.min(), X.max(), y.min(), y.max()))
    return X, y

def loadDataGeneral_multiscale(df, path,img_size):
    """
    This function loads data stored in nifti format. Data should already be of
    appropriate shape.

    Inputs:
    - df: Pandas dataframe with two columns: image filenames and ground truth filenames.
    - path: Path to folder containing filenames from df.
    - append_coords: Whether to append coordinate channels or not.
    Returns:
    - X: Array of 3D images with 1 or 4 channels depending on `append_coords`.
    - y: Array of 3D masks with 1 channel.
    """
    X, y, X2 = [], [], []
    for i, item in df.iterrows():
        img = nib.load(path + item[0]).get_data()
        mask = nib.load(path + item[1]).get_data()
        img2 = nib.load(path + item[2]).get_data()
        mask = np.clip(mask, 0, 255)
        out = mask
        img = img#*1./33240
        img = scipy.ndimage.zoom(img, [img_size/img.shape[0],img_size/img.shape[1], 1], order=1)
        out = scipy.ndimage.zoom(out, [img_size / out.shape[0], img_size / out.shape[1], 1], order=1)

        X.append(img)
        y.append(out)
        X2.append(img2)
        #cmask = (mask * 1. / 255)

    X = np.expand_dims(X, -1)
    y = np.expand_dims(y, -1)
    X2 = np.expand_dims(X2, -1)
    #y = np.concatenate((1 - y, y), -1)
    #y = np.array(y)
    print('### Dataset loaded')
    print('\t{}'.format(path))
    #print('\t{}\t{}'.format(X.shape, y.shape))
    #print('\tX:{:.1f}-{:.1f}\ty:{:.1f}-{:.1f}\n'.format(X.min(), X.max(), y.min(), y.max()))
    return X, y, X2