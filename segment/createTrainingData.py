import glob as glob
import os

import numpy as np
import tifffile
import nibabel as nib
import re
import csv
import skimage


def niftiwriteF(a, b):
    nib.save(nib.Nifti1Image(a, affine=np.eye(4)), b)


def niftiwrite8(a, b):
    nib.save(nib.Nifti1Image(np.uint8(a), affine=np.eye(4)), b)


def generateTrainData(trainingFiles,tfolder):
    sizeI = (32, 32)

    allfiles = trainingFiles
    print(allfiles)
    tfolder = tfolder
    if not os.path.isdir(tfolder):
        os.makedirs(tfolder)

    
    class_dict_csv = tfolder + '/' + 'class_dict.csv'
    with open(class_dict_csv,'w') as cd:
        writer = csv.writer(cd)
        writer.writerow(['name','r','g','b'])
        writer.writerow(['Cell',255,255,255])
        writer.writerow(['Background',0,0,0])
        cd.close()

    # vfolder = os.path.dirname(allfiles[0])
    # vfolder += '/'+ os.path.basename(allfiles[0])[0:4]+'ValidData'

    idx_train = tfolder + '/' + 'idx_train.csv'
    print(idx_train)
    # idx_valid = vfolder + '/' + 'idx_valid.csv'

    with open(idx_train, 'w') as idxtrain:
        writer = csv.writer(idxtrain)
        writer.writerow(['path', 'pathmsk'])
        idxtrain.close()

    for file_i in allfiles:
        filename = os.path.basename(file_i).split('.')[0]
        fname = re.sub(r'\W+', '', filename)

        idx_train = tfolder + '/' + 'idx_train.csv'
        print(idx_train)
        # idx_valid = vfolder + '/' + 'idx_valid.csv'

        # with open(idx_valid,'w') as idxvalid:
        #     writer2 = csv.writer(idxvalid)
        #     writer2.writerow('path', 'pathmsk')

        fnameT = 'trainimg_' + fname + '_'
        fnameGT = 'trainimg_' + fname + '_GT_'

        Tdata = np.zeros((sizeI[0], sizeI[1]))
        Gdata = np.zeros((sizeI[0], sizeI[1]))
        data = tifffile.imread(file_i)

        if (np.size(np.shape(data)) > 3 and np.size(data, 1) < 3):
            data = np.transpose(data, (3, 2, 0, 1))

        # if (np.size(data, 2) == 13):
        #     data_i = np.zeros((512,320,15,2))
        #     data_i[:,:,1:14,:] = data
        #     data_i[:,:,0,:] = data_i[:,:,1,:]
        #     data_i[:,:,14,:] = data_i[:,:,13,:]

        
        dataT = data[:, :, :, 0].squeeze()
        dataG = data[:, :, :, 1].squeeze()

        count = 0
        for i in range(0, int(np.size(dataT, 0) / sizeI[0])):
            for j in range(0, int(np.size(dataT, 1) / sizeI[1])):
                count += 1
                Tdata = dataT[i * sizeI[0]:(i + 1) * sizeI[0], j * sizeI[1]:(j + 1) * sizeI[1]]
                Gdata = dataG[i * sizeI[0]:(i + 1) * sizeI[0], j * sizeI[1]:(j + 1) * sizeI[1]]
                Tdata = (Tdata - np.min(Tdata)) / (np.max(Tdata) - (np.min(Tdata))) * 0.5 + 0.5
                Gdata = np.uint8(Gdata)
                ext = '00' if count < 10 else '0' if count < 100 else ''
                path = fnameT + ext + str(count) + '.nii'
                pathGT = fnameGT + ext + str(count) + '.nii'
                niftiwriteF(Tdata, tfolder + '/' + path)
                niftiwrite8(Gdata, tfolder + '/' + pathGT)

                with open(idx_train, 'a') as file:
                    writer = csv.writer(file)
                    writer.writerow([path, pathGT])
                    file.close()
    


def generateValidationData(validationFiles,vfolder):
    sizeI = (32, 32)

    allfiles = validationFiles
    print(allfiles)
    vfolder = vfolder
    if not os.path.isdir(vfolder):
        os.makedirs(vfolder)
    # vfolder = os.path.dirname(allfiles[0])
    # vfolder += '/'+ os.path.basename(allfiles[0])[0:4]+'ValidData'

    idx_valid = vfolder + '/' + 'idx_val.csv'
    print(idx_valid)
    # idx_valid = vfolder + '/' + 'idx_valid.csv'

    with open(idx_valid, 'w') as idxvalid:
        writer = csv.writer(idxvalid)
        writer.writerow(['path', 'pathmsk'])
        idxvalid.close()

    for file_i in allfiles:
        filename = os.path.basename(file_i).split('.')[0]
        fname = re.sub(r'\W+', '', filename)

        idx_valid = vfolder + '/' + 'idx_val.csv'
        print(idx_valid)

        fnameV = 'validimg_' + fname + '_'
        fnameGT = 'validimg_' + fname + '_GT_'

        Vdata = np.zeros((sizeI[0], sizeI[1]))
        Gdata = np.zeros((sizeI[0], sizeI[1]))
        data = tifffile.imread(file_i)
        

        if (np.size(np.shape(data)) > 3 and np.size(data, 1) < 3):
            data = np.transpose(data, (3, 2, 0, 1))

        # if (np.size(data, 2) == 13):
        #     data_i = np.zeros((512,320,15,2))
        #     data_i[:,:,1:14,:] = data
        #     data_i[:,:,0,:] = data_i[:,:,1,:]
        #     data_i[:,:,14,:] = data_i[:,:,13,:]

    
        dataV = data[:, :, :, 0].squeeze()
        dataG = data[:, :, :, 1].squeeze()


        count = 0
        for i in range(0, int(np.size(dataV, 0) / sizeI[0])):
            for j in range(0, int(np.size(dataV, 1) / sizeI[1])):
                count += 1
                Vdata = dataV[i * sizeI[0]:(i + 1) * sizeI[0], j * sizeI[1]:(j + 1) * sizeI[1]]
                Gdata = dataG[i * sizeI[0]:(i + 1) * sizeI[0], j * sizeI[1]:(j + 1) * sizeI[1]]
                Vdata = (Vdata - np.min(Vdata)) / (np.max(Vdata) - (np.min(Vdata))) * 0.5 + 0.5
                Gdata = np.uint8(Gdata)
                ext = '00' if count < 10 else '0' if count < 100 else ''
                path = fnameV + ext + str(count) + '.nii'
                pathGT = fnameGT + ext + str(count) + '.nii'
                niftiwriteF(Vdata, vfolder + '/' + path)
                gsave=nib.Nifti1Image(Gdata, affine=np.eye(4))
                nib.save(gsave,vfolder + '/' + pathGT)

                with open(idx_valid, 'a') as file:
                    writer = csv.writer(file)
                    writer.writerow([path, pathGT])
                    file.close()

def create(addr):
    Files1 = sorted(glob.glob(addr + '/' + '*.tif'))
    trainingFiles = Files1[:-2]
    validationFiles = Files1[-2:]

    tfolder = os.path.dirname(trainingFiles[0])
    tfolder += '/' + os.path.basename(trainingFiles[0])[0:5] + 'TrainData'

    vfolder = os.path.dirname(trainingFiles[0])
    vfolder += '/' + os.path.basename(trainingFiles[0])[0:5] + 'ValidData'

    generateTrainData(trainingFiles,tfolder)
    generateValidationData(validationFiles,vfolder)
    return tfolder, vfolder


# generateTrainData('/home/nirvan/Desktop/ATdata')
# generateValidationData('/home/nirvan/Desktop/AVdata')
