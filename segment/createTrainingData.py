import glob as glob
import os

import numpy as np
import tifffile
import nibabel as nib
import re
import csv
import random
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

def create(addr,continue_training=False,same_model=False):
    if not os.path.isdir(addr + '/new'):
        os.makedirs(addr + '/new')

    Files = sorted(glob.glob(addr + '/'+'*.tif'))
    count = len(Files)



    prename = os.path.basename(Files[0])[0:9]
    print(prename)

    for a, file in enumerate(Files):

        if(a<1 and not continue_training):
            V_sample = tifffile.imread(file)

            dims = np.shape(V_sample)

            ## Create shuffled versions of the images
            FinalImage = np.zeros(np.shape(V_sample))

            images = []

            for i in range(0, int(dims[-1] / 8)):
                for j in range(0, int(dims[-2] / 8)):
                    im = V_sample[:, :, j * 8:(j + 1) * 8, i * 8:(i + 1) * 8]
                    images.append(im)

            random.shuffle(images)
            num = 0
            for i in range(0, int(dims[-1] / 8)):
                for j in range(0, int(dims[-2] / 8)):
                    FinalImage[:, :, j * 8:(j + 1) * 8, i * 8:(i + 1) * 8] = images[num]
                    num += 1
            FinalImage = np.uint16(FinalImage)

            ## Filter tiny fragments
            # Select the C=1 channel
            channel = 1
            data_c1 = FinalImage[:, channel, :, :]

            # Create a binary image of pixels greater than 0
            binary_image = (data_c1 > 0)

            # Label connected components
            labeled_image = skimage.measure.label(binary_image)
            # Calculate size of each component
            component_sizes = np.bincount(labeled_image.ravel())
            ############################################################
            object_labels = np.unique(labeled_image)[1:]
            tinylist=[]

            for label in object_labels:
                # find coordinates of object in labeled image
                object_coords = np.where(labeled_image == label)
                # calculate number of voxels in object
                num_voxels = len(object_coords[0])
                # calculate depth in z direction
                z_min = np.min(object_coords[0])
                z_max = np.max(object_coords[0])
                z_depth = z_max - z_min + 1
                if (num_voxels <= 5 or (z_depth == 1 and num_voxels <= 8 and z_max != dims[0] - 1)) :
                    tinylist.append(label)

            for star in tinylist:
                    labeled_image[labeled_image == star] = 0

            # # Remove components smaller than 9 pixels
            # min_size = 9
            # remove_mask = component_sizes < min_size
            # remove_mask[0] = 0  # Do not remove background
            # # no need to label here, I just copied my code from tracking
            # labeled_image[remove_mask[labeled_image]] = 0
            #######################################################
            labeled_image = labeled_image.astype('uint16')
            labeled_image[labeled_image > 0] = 65535
            # Create a copy of the original data with C=0 channel unchanged
            data_c0 = np.copy(FinalImage[:, 0, :, :])

            # Save the output TIFF file with C=0 channel unchanged
            FinalImage = np.stack((data_c0, labeled_image), axis=1)
            ###

            tifffile.imwrite(addr + '/new/' + prename + 't_orig_' + str(a) + '.tif', V_sample, imagej=True)
            tifffile.imwrite(addr + '/new/' + prename + 't_ashuffled' + str(a) + '.tif', FinalImage, imagej=True)
        elif not same_model:
            V_sample = tifffile.imread(file)
            tifffile.imwrite(addr + '/new/' + prename + 't_orig_' + str(a) + '.tif', V_sample, imagej=True)


    if not continue_training:
        Files1 = sorted(glob.glob(addr + '/new/' + '*.tif'))
        trainingFiles = Files1[:-2]
        validationFiles = Files1[-2:]

        tfolder = os.path.dirname(trainingFiles[0])
        tfolder += '/' + prename + 'TrainData'

        vfolder = os.path.dirname(trainingFiles[0])
        vfolder += '/' + prename + 'ValidData'

        generateTrainData(trainingFiles,tfolder)
        generateValidationData(validationFiles,vfolder)
        return tfolder, vfolder

    elif not same_model:
        Files1 = sorted(glob.glob(addr + '/new/' + '*.tif'))
        trainingFiles = Files1[:-1]
        validationFiles = Files1[-1:]

        tfolder = os.path.dirname(trainingFiles[0])
        tfolder += '/' + prename + 'TrainData'

        vfolder = os.path.dirname(trainingFiles[0])
        vfolder += '/' + prename + 'ValidData'

        generateTrainData(trainingFiles, tfolder)
        generateValidationData(validationFiles, vfolder)
        return tfolder, vfolder

    elif same_model:
        Files1 = sorted(glob.glob(addr + '/new/' + '*.tif'))
        trainingFiles = Files1[:-2]
        validationFiles = Files1[-2:]

        tfolder = os.path.dirname(trainingFiles[0])
        tfolder += '/' + prename + 'TrainData'

        vfolder = os.path.dirname(trainingFiles[0])
        vfolder += '/' + prename + 'ValidData'
        return tfolder,vfolder
