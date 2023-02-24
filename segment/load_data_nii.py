import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.ndimage
def loadDataGeneral(path,img_size):
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
    X, y= [], []
    for file in path:
        img = nib.load(file).get_data()
        size=img.shape
        img1 = img[:, 0:int(size[1]/2), :]
        img2 = img[:, int(size[1] / 2):int(size[1]), :]
        # img1 = scipy.ndimage.zoom(img1, [img_size[0]/img.shape[0],img_size[1]/img.shape[1], 1], order=1)
        # img2 = scipy.ndimage.zoom(img2, [img_size[0] / img.shape[0], img_size[1] / img.shape[1], 1], order=1)


        X.append(np.float32(img1))
        y.append(np.float32(img2))
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

def loadDataGeneral_test(path,img_size):
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
    X= []
    for file in path:
        img = nib.load(file).get_data()
        size=img.shape
        img1 = img[:, 0:int(size[1]), :]
        # img1 = scipy.ndimage.zoom(img1, [img_size[0]/img.shape[0],img_size[1]/img.shape[1], 1], order=1)
        # img2 = scipy.ndimage.zoom(img2, [img_size[0] / img.shape[0], img_size[1] / img.shape[1], 1], order=1)


        X.append(np.float32(img1))

    X = np.expand_dims(X, -1)
    #y = np.concatenate((1 - y, y), -1)
    #y = np.array(y)
    print('### Dataset loaded')
    print('\t{}'.format(path))
    #print('\t{}\t{}'.format(X.shape, y.shape))
    #print('\tX:{:.1f}-{:.1f}\ty:{:.1f}-{:.1f}\n'.format(X.min(), X.max(), y.min(), y.max()))
    return X

def loadDataGeneral_multiscale(path,img_size):
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
    X, y= [], []
    for file in path:
        img = nib.load(file).get_data()
        size=img.shape
        print(size)
        stop
        img1 = img[:, 0:int(size[1]/2), :]
        img2 = img[:, int(size[1] / 2):int(size[1]), :]
        # img1 = scipy.ndimage.zoom(img1, [img_size[0]/img.shape[0],img_size[1]/img.shape[1], 1], order=1)
        # img2 = scipy.ndimage.zoom(img2, [img_size[0] / img.shape[0], img_size[1] / img.shape[1], 1], order=1)


        X.append(np.float32(img1))
        y.append(np.float32(img2))
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
