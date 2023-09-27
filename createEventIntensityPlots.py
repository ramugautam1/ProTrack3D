import os.path

import numpy as np
import nibabel as nib
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import matplotlib as mpl
from functions import starline


def niftireadU32(arg):
    return np.asarray(nib.load(arg).dataobj).astype(np.uint32).squeeze()


def createEventsAndIntensityPlots(segpath, filePath, modelName, startpoint, endpoint, originalImage,sT,eT,sTime):
    eventsAndIntensityDF = pd.DataFrame(columns=['time', 'total_intensity', 'masked_intensity', 'pixel_count',  # done
                                                 'masked_intensity_per_pixel', 'masked_intensity_per_object',
                                                 'total_intensity_per_pixel', 'total_intensity_per_object',  # done
                                                 'total_objects', 'split', 'merge', 'new_id', 'retired_id', 'birth',
                                                 'death',  # done
                                                 'split_rate', 'merge_rate', 'birth_rate', 'death_rate',  # done
                                                 'MA_total_intensity', 'MA_masked_intensity', 'MA_pixel_count',
                                                 'MA_total_objects',  # done
                                                 'MA_masked_intensity_per_pixel', 'MA_masked_intensity_per_object',  # done
                                                 'MA_total_objects', 'MA_split', 'MA_merge', 'MA_new_id', 'MA_retired_id',
                                                 'MA_birth', 'MA_death',  # done
                                                 'MA_split_rate', 'MA_merge_rate', 'MA_birth_rate',
                                                 'MA_death_rate'])  # done

    file_path = filePath
    eventsDF = pd.read_csv(file_path + '/all_events.csv')

    time_ = [int(i + 1) for i in range(len(eventsDF))]
    eventsAndIntensityDF['time'] = time_
    eventsAndIntensityDF['split'] = eventsDF['Split'].astype('int')
    eventsAndIntensityDF['merge'] = eventsDF['Merge'].astype('int')
    eventsAndIntensityDF['new_id'] = eventsDF['new id'].astype('int')
    eventsAndIntensityDF['retired_id'] = eventsDF['retired id'].astype('int')
    eventsAndIntensityDF['birth'] = eventsDF['birth'].astype('int')
    eventsAndIntensityDF['death'] = eventsDF['death'].astype('int')
    eventsAndIntensityDF['total_objects'] = eventsDF['Total objects'].astype('int')

    mask = niftireadU32(segpath + 'CombinedSO/CombinedSO.nii')[:,:,:,:]

    print('mask shape:  ', mask.shape)
    newMask = np.zeros_like(mask)
    newMask[mask > 0] = 1
    print('new mask shape:  ', newMask.shape)

    image = tifffile.imread(originalImage)
    image = np.transpose(image, (3, 2, 1, 0))
    image = image[:, :, :, sT:eT]

    print('image shape:  ', image.shape)

    maskedImage = image * newMask
    print(np.shape(image))

    distance=endpoint-startpoint+1

    # print(eT, sT, eT-sT-1, startpoint, endpoint, endpoint-startpoint, distance)
    print('--------------------------------------------------------------------------------------------------------')
    intensityArr = np.zeros((len(eventsDF), 3))
    # for i in range(np.size(image, 3)):
    print("'d,l,i3,mi3,nm3'  -1   :",distance-1,len(eventsDF)-1, image.shape[3]-1, maskedImage.shape[3]-1, newMask.shape[3]-1)
    for i in range(min(distance-1, len(eventsDF)-1, image.shape[3]-1, maskedImage.shape[3]-1, newMask.shape[3]-1)):
        intensityArr[i, :] = [image[:, :, :, i].sum(), maskedImage[:, :, :, i].sum(), newMask[:, :, :, i].sum()]
    intensityDF = pd.DataFrame(intensityArr, columns=['total_intensity', 'masked_intensity', 'pixel_count'])
    for column in intensityDF.columns:
        eventsAndIntensityDF[column] = intensityDF[column].astype('int')

    eventsAndIntensityDF['masked_intensity_per_pixel'] = eventsAndIntensityDF['masked_intensity'] / eventsAndIntensityDF['pixel_count']
    eventsAndIntensityDF['masked_intensity_per_object'] = eventsAndIntensityDF['masked_intensity'] / eventsAndIntensityDF[
        'total_objects']
    eventsAndIntensityDF['total_intensity_per_pixel'] = eventsAndIntensityDF['total_intensity'] / eventsAndIntensityDF[
        'pixel_count']
    eventsAndIntensityDF['total_intensity_per_object'] = eventsAndIntensityDF['total_intensity'] / eventsAndIntensityDF[
        'total_objects']

    eventsAndIntensityDF['split_rate'] = eventsAndIntensityDF['split'] / eventsAndIntensityDF['total_objects'].shift(1)
    eventsAndIntensityDF['merge_rate'] = eventsAndIntensityDF['merge'] / eventsAndIntensityDF['total_objects'].shift(1)
    eventsAndIntensityDF['birth_rate'] = eventsAndIntensityDF['birth'] / eventsAndIntensityDF['total_objects'].shift(1)
    eventsAndIntensityDF['death_rate'] = eventsAndIntensityDF['death'] / eventsAndIntensityDF['total_objects'].shift(1)

    for r, m in [('total_objects', 'MA_total_objects'),
                 ('total_intensity', 'MA_total_intensity'),
                 ('masked_intensity', 'MA_masked_intensity'),
                 ('pixel_count', 'MA_pixel_count'),
                 ('total_objects', 'MA_total_objects'),
                 ('masked_intensity_per_pixel', 'MA_masked_intensity_per_pixel'),
                 ('masked_intensity_per_object', 'MA_masked_intensity_per_object'),
                 ('split', 'MA_split'), ('merge', 'MA_merge'),
                 ('new_id', 'MA_new_id'), ('retired_id', 'MA_retired_id'),
                 ('birth', 'MA_birth'), ('death', 'MA_death'),
                 ('split_rate', 'MA_split_rate'), ('merge_rate', 'MA_merge_rate'),
                 ('birth_rate', 'MA_birth_rate'),
                 ('death_rate', 'MA_death_rate')]:
        eventsAndIntensityDF[m] = eventsAndIntensityDF[r].rolling(window=9, center=True).mean()

    fig, axs = plt.subplots(3, 3, figsize=(40, 20))
    axs[0, 0].plot(eventsAndIntensityDF['MA_total_intensity'][:-4])
    axs[0, 0].set_title('Total Intensity')
    axs[0, 0].legend(labels=['Total Intensity'])
    axs[0, 1].plot(eventsAndIntensityDF['MA_masked_intensity'][:-4])
    axs[0, 1].set_title('Masked Intensity')
    axs[0, 1].legend(labels=['Masked Intensity'])
    axs[0, 2].plot(eventsAndIntensityDF['MA_total_objects'][:-4])
    axs[0, 2].set_title('Total Objects')
    axs[0, 2].legend(labels=['Total Objects'])

    axs[1, 0].plot(eventsAndIntensityDF['MA_masked_intensity_per_object'][:-4])
    axs[1, 0].set_title('Masked Intensity Per Object')
    axs[1, 0].legend(labels=['Masked Intensity Per Object'])
    axs[1, 1].plot(eventsAndIntensityDF['MA_masked_intensity_per_pixel'][:-4])
    axs[1, 1].set_title('Masked Intensity Per Pixel')
    axs[1, 1].legend(labels=['Masked Intensity Per Pixel'])

    axs[1, 2].plot(eventsAndIntensityDF['MA_split'][:-4])
    axs[1, 2].plot(eventsAndIntensityDF['MA_merge'][:-4])
    axs[1, 2].set_title('Split and Merge')
    axs[1, 2].legend(labels=['Split', 'Merge'])

    axs[2, 0].plot(eventsAndIntensityDF['MA_birth'][:-4])
    axs[2, 0].plot(eventsAndIntensityDF['MA_death'][:-4])
    axs[2, 0].set_title('Birth and Death')
    axs[2, 0].legend(labels=['Birth', 'Death'])

    axs[2, 1].plot(eventsAndIntensityDF['MA_split_rate'][:-4])
    axs[2, 1].plot(eventsAndIntensityDF['MA_merge_rate'][:-4])
    axs[2, 1].set_title('Split Rate and Merge Rate')
    axs[2, 1].legend(labels=['Split Rate', 'Merge Rate'])

    axs[2, 2].plot(eventsAndIntensityDF['MA_birth_rate'][:-4])
    axs[2, 2].plot(eventsAndIntensityDF['MA_death_rate'][:-4])
    axs[2, 2].set_title('Birth Rate and Death Rate')
    axs[2, 2].legend(labels=['Birth Rate', 'Death Rate'])

    nameOnly = os.path.basename(originalImage)[:-4]

    plt.savefig(file_path + '/' + 'EventsAndIntensityPlots_' + nameOnly + '.png')
    eventsAndIntensityDF.to_csv(file_path + '/' + 'EventsAndIntensityPlotsData_' + nameOnly + '.csv')

    import time as theTime
    nowtime = theTime.perf_counter()
    totalTimeTaken = nowtime-sTime
    starline()
    hours_ = int(totalTimeTaken//60//60)
    min_ = int((totalTimeTaken -  hours_ * 60*60)//60)
    print(f'Tracking Complete. Total Time: {hours} hours {min_} min {int(totalTimeTaken%60 + 1)} sec ')
    starline()