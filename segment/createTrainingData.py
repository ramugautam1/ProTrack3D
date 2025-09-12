import glob as glob
import os

import numpy as np
import tifffile
import nibabel as nib
import re
import csv
import random
import skimage
import shutil
from datetime import datetime


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

# # def create(addr,continue_training=False,same_model=False):
# def create(addr, transfer_learning=False, continue_training=False):
#     trainingdatalist=[]
#     validationdatalist=[]
#     time_tag = datetime.now().strftime("%m%d%Y%H%M%S")
#     newdir = addr + '/aug' + time_tag
#
#     if not os.path.isdir(newdir):
#         os.makedirs(newdir)
#
#     try:
#         tif_files = [f for f in os.listdir(addr) if f.endswith('.tif')]
#
#         for tif_file in tif_files:
#             source_path = os.path.join(addr, tif_file)
#             destination_path = os.path.join(newdir, tif_file)
#             shutil.copy2(source_path, destination_path)
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#
#     if not continue_training:
#         gt_path = newdir
#         files = os.listdir(gt_path)
#         # Sort the files by modification time (most recent first)
#         files.sort(key=lambda x: os.path.getmtime(os.path.join(gt_path, x)))
#         # Iterate through the files and rename them
#         valsize = 2 if len(files) > 5 else 1
#         for i, filename in enumerate(files):
#             if i >= len(files) - valsize:
#                 # This is the last file, add 'v' prefix
#                 new_name = 'v_' + filename[:-4] + time_tag + '.tif'
#             else:
#                 # Add 't' prefix to all other files
#                 new_name = 't_' + filename[:-4] + time_tag + '.tif'
#             # Create the new file path
#             old_path = os.path.join(gt_path, filename)
#             new_path = os.path.join(gt_path, new_name)
#             # Rename the file
#             os.rename(old_path, new_path)
#         Files = glob.glob(os.path.join(gt_path, '*.tif'))
#
#         for file in Files:
#             if os.path.basename(file)[0:2] == 't_':
#                 imr = tifffile.imread(file)
#                 # save_name_o = os.path.dirname(file)  + os.path.basename(file)
#                 # tifffile.imsave(save_name_o, imr, imagej=True, dtype = imr.dtype)
#
#                 dtype_ = type(imr[0, 0, 0, 0])
#                 # print(dtype_)
#                 imrs = np.flip(imr, axis=(-2, -1))
#                 imrss = np.flip(imr, axis=-2)
#
#                 save_name_f1 = file[:-4] + '_f1.tif'
#                 save_name_f12 = file[:-4] + '_f12.tif'
#
#                 # save_name_f1 = os.path.dirname(file) + '/aug/' + os.path.basename(file)[:-4] + '_f1.tif'
#                 # save_name_f12 = os.path.dirname(file) + '/aug/' + os.path.basename(file)[:-4] + '_f12.tif'
#
#                 img_channel = imr[:, 0, :, :]
#
#                 tifffile.imsave(save_name_f1, imrs, imagej=True, dtype=img_channel.dtype)
#                 tifffile.imsave(save_name_f12, imrss, imagej=True, dtype=img_channel.dtype)
#
#
#                 noise_density = 0.0005
#                 salt_value = 65535  # Maximum value for uint16
#                 pepper_value = 0
#
#                 salt_mask = np.random.rand(*img_channel.shape) < noise_density
#                 img_channel[salt_mask] = salt_value
#
#                 imr[:, 0, :, :] = img_channel
#                 save_name_sp = file[:-4] + '_sp.tif'
#                 tifffile.imsave(save_name_sp, imr, imagej=True, dtype=img_channel.dtype)
#                 # save_name_sp = os.path.dirname(file) + '/aug/' + os.path.basename(file)[:-4] + '_sp.tif'
#
#                 noise_density=0.001
#                 img_channel[salt_mask] = pepper_value
#                 imr[:, 0, :, :] = img_channel
#                 save_name_pp = file[:-4] + '_pp.tif'
#
#                 tifffile.imsave(save_name_pp, imr, imagej=True, dtype=img_channel.dtype)
#
#         Files = sorted(glob.glob(gt_path + '/*.tif'))
#
#         for a, file in enumerate(Files):
#
#             if ((a < 1 or a==6) and os.path.basename(file)[:2]=='t_'): # and not transfer_learning):
#                 V_sample = tifffile.imread(file)
#
#                 dims = np.shape(V_sample)
#
#                 ## Create shuffled versions of the images
#                 FinalImage = np.zeros(np.shape(V_sample))
#
#                 images = []
#
#                 for i in range(0, int(dims[-1] / 8)):
#                     for j in range(0, int(dims[-2] / 8)):
#                         im = V_sample[:, :, j * 8:(j + 1) * 8, i * 8:(i + 1) * 8]
#                         images.append(im)
#
#                 random.shuffle(images)
#                 num = 0
#                 for i in range(0, int(dims[-1] / 8)):
#                     for j in range(0, int(dims[-2] / 8)):
#                         FinalImage[:, :, j * 8:(j + 1) * 8, i * 8:(i + 1) * 8] = images[num]
#                         num += 1
#                 FinalImage = np.uint16(FinalImage)
#
#                 ## Filter tiny fragments
#                 # Select the C=1 channel
#                 channel = 1
#                 data_c1 = FinalImage[:, channel, :, :]
#
#                 # Create a binary image of pixels greater than 0
#                 binary_image = (data_c1 > 0)
#
#                 # Label connected components
#                 labeled_image = skimage.measure.label(binary_image)
#                 # Calculate size of each component
#                 component_sizes = np.bincount(labeled_image.ravel())
#                 ############################################################
#                 object_labels = np.unique(labeled_image)[1:]
#                 tinylist = []
#
#                 for label in object_labels:
#                     # find coordinates of object in labeled image
#                     object_coords = np.where(labeled_image == label)
#                     # calculate number of voxels in object
#                     num_voxels = len(object_coords[0])
#                     # calculate depth in z direction
#                     z_min = np.min(object_coords[0])
#                     z_max = np.max(object_coords[0])
#                     z_depth = z_max - z_min + 1
#                     if (num_voxels <= 5 or (z_depth == 1 and num_voxels <= 8 and z_max != dims[0] - 1)):
#                         tinylist.append(label)
#
#                 for star in tinylist:
#                     labeled_image[labeled_image == star] = 0
#
#                 labeled_image = labeled_image.astype('uint16')
#                 labeled_image[labeled_image > 0] = 65535
#                 # Create a copy of the original data with C=0 channel unchanged
#                 data_c0 = np.copy(FinalImage[:, 0, :, :])
#
#                 # Save the output TIFF file with C=0 channel unchanged
#                 FinalImage = np.stack((data_c0, labeled_image), axis=1)
#                 ###
#                 # filenameshuffle = os.path.dirname(file) + '/aug/' + os.path.basename(file)[:-4] + 'shuf.tif'
#                 tifffile.imwrite(file[:-4] +  'shuf.tif', V_sample, imagej=True)
#
#         for file in Files:
#             if os.path.basename(file)[0:2] == 't_':
#                 trainingdatalist.append(file)
#             elif os.path.basename(file)[0:2] == 'v_':
#                 validationdatalist.append(file)
#
#         Files = sorted(glob.glob(gt_path + '/*.tif'))
#         print(len(Files))
#
#         dataName = os.path.basename(Files[0])[2:-4]
#         tfolder = os.path.dirname(Files[0]) + '/' + dataName + 'T'
#         vfolder = os.path.dirname(Files[0]) + '/' + dataName + 'V'
#         generateTrainData(trainingdatalist, tfolder)
#         generateValidationData(validationdatalist, vfolder)
#         return tfolder, vfolder

def cutAndDry(addr, newdir):
    try:
        tif_files = [f for f in os.listdir(addr) if f.endswith('.tif')]
        tifcount = len(tif_files)
        tifVcount = int(np.ceil(tifcount*0.15))
        tifTcount = tifcount - tifVcount
        tile_size = 32;
        step = 14;
        tile_id=0;

        savename = tif_files[0][:-4]

        for i,tif_file in enumerate(tif_files):
            if i<tifTcount:
                source_path = os.path.join(addr, tif_file)

                tifIn = tifffile.imread(source_path)
                tifInFlip = tifIn[::-1,:,:,:]
                Z,C,H,W = tifIn.shape
                for y in range(0, H-tile_size + 1, step):
                    for x in range(0, W-tile_size+1, step):
                        tile = tifIn[:,:,y:y+tile_size, x:x+tile_size]
                        tileF = tifInFlip[:,:,y:y+tile_size, x:x+tile_size]
                        tn = f'{tile_id:04d}'
                        tnf = f'{(tile_id+1):04d}'
                        filename = f't_{savename}_{tn}.tif'
                        filenameF = f't_{savename}_{tnf}.tif'
                        filepath = os.path.join(newdir, filename)
                        filepathF = os.path.join(newdir, filenameF)

                        tifffile.imwrite(filepath, tile, imagej=True, metadata={'axes':'ZCYX'})
                        tifffile.imwrite(filepathF, tileF, imagej=True, metadata={'axes': 'ZCYX'})

                        tile_id +=2
            else:
                source_path = os.path.join(addr, tif_file)
                destination_path = os.path.join(newdir, 'v_' + tif_file)

                tifIn = tifffile.imread(source_path)
                tifInFlip = tifIn[::-1, :, :, :]
                Z, C, H, W = tifIn.shape
                for y in range(0, H - tile_size + 1, step):
                    for x in range(0, W - tile_size + 1, step):
                        tile = tifIn[:, :, y:y + tile_size, x:x + tile_size]
                        tileF = tifInFlip[:, :, y:y + tile_size, x:x + tile_size]
                        tn = f'{tile_id:04d}.tif'
                        tnf = f'{(tile_id + 1):04d}.tif'
                        filename = f'v_{savename}_{tn}.tif'
                        filenameF = f'v_{savename}_{tnf}.tif'
                        filepath = os.path.join(newdir, filename)
                        filepathF = os.path.join(newdir, filenameF)

                        tifffile.imwrite(filepath, tile, imagej=True, metadata={'axes': 'ZCYX'})
                        tifffile.imwrite(filepathF, tileF, imagej=True, metadata={'axes': 'ZCYX'})

                        tile_id += 2
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def create(addr, transfer_learning=False, continue_training=False):
    trainingdatalist = []
    validationdatalist = []
    time_tag = datetime.now().strftime("%m%d%Y%H%M%S")
    newdir = addr + '/aug' #+ time_tag

    if not os.path.isdir(newdir):
        os.makedirs(newdir)

    cutAndDry(addr, newdir)

    if not continue_training:
        gt_path = newdir
        files = os.listdir(newdir)
        # Sort the files by modification time (most recent first)
        files.sort(key=lambda x: os.path.getmtime(os.path.join(gt_path, x)))

        files = sorted(glob.glob(gt_path + '/*.tif'))

        for file in files:
            if os.path.basename(file)[0:2] == 't_':
                trainingdatalist.append(file)
            elif os.path.basename(file)[0:2] == 'v_':
                validationdatalist.append(file)

        Files = sorted(glob.glob(gt_path + '/*.tif'))
        print(len(Files))

        dataName = os.path.basename(Files[0])[2:-4]
        tfolder = os.path.dirname(Files[0]) + '/' + dataName + 'T'
        vfolder = os.path.dirname(Files[0]) + '/' + dataName + 'V'
        generateTrainData(trainingdatalist, tfolder)
        generateValidationData(validationdatalist, vfolder)
        return tfolder, vfolder

    elif continue_training:
        allaugfolders = [folder for folder in os.listdir(addr) if
                            folder.startswith('aug') and os.path.isdir(os.path.join(addr, folder))]
        augfolder = allaugfolders[0]
        Files = sorted(glob.glob(addr + '/' + augfolder + '/*.tif'))
        dataName = os.path.basename(Files[0])[2:-4]
        tfolder = addr + '/' + augfolder + '/' + dataName + 'T'
        vfolder = addr + '/' + augfolder + '/' + dataName + 'V'

        return tfolder, vfolder



    #
    # if continue_training:
    #     Files = sorted(glob.glob(addr + '/' + '*.tif'))
    #
    #     trainingFiles = Files[:-2]
    #     validationFiles = Files[-2:]
    #
    #     tfolder = os.path.dirname(Files[0]) + '/' + dataName + 'TrainData'
    #     vfolder = os.path.dirname(Files[0]) + '/' + dataName + 'ValidData'
    #
    #
    #     generateTrainData(trainingFiles,tfolder)
    #     generateValidationData(validationFiles,vfolder)
    #     return tfolder, vfolder
    #
    # elif not same_model:
    #     Files1 = sorted(glob.glob(addr + '/new/' + '*.tif'))
    #     trainingFiles = Files1[:-1]
    #     validationFiles = Files1[-1:]
    #
    #     tfolder = os.path.dirname(trainingFiles[0])
    #     tfolder += '/' + prename + 'TrainData'
    #
    #     vfolder = os.path.dirname(trainingFiles[0])
    #     vfolder += '/' + prename + 'ValidData'
    #
    #     generateTrainData(trainingFiles, tfolder)
    #     generateValidationData(validationFiles, vfolder)
    #     return tfolder, vfolder
    #
    # elif same_model:
    #     Files1 = sorted(glob.glob(addr + '/' + '*.tif'))
    #     dataName = os.path.basename(Files[0])[:-4]
    #     tfolder = os.path.dirname(Files[0]) + '/' + dataName + 'TrainData'
    #     vfolder = os.path.dirname(Files[0]) + '/' + dataName + 'ValidData'
    #
    #     return tfolder,vfolder
