import os.path

import numpy as np
import nibabel as nib
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import matplotlib as mpl
from functions import starline

#
# def niftireadU32(arg):
#     return np.asarray(nib.load(arg).dataobj).astype(np.uint32).squeeze()
#
#
# def createEventsAndIntensityPlots(segpath, filePath, modelName, startpoint, endpoint, originalImage,sT,eT,sTime):
#     eventsAndIntensityDF = pd.DataFrame(columns=['time', 'total_intensity', 'masked_intensity', 'pixel_count',  # done
#                                                  'masked_intensity_per_pixel', 'masked_intensity_per_object',
#                                                  'total_intensity_per_pixel', 'total_intensity_per_object',  # done
#                                                  'total_objects', 'split', 'merge', 'new_id', 'retired_id', 'birth',
#                                                  'death',  # done
#                                                  'split_rate', 'merge_rate', 'birth_rate', 'death_rate',  # done
#                                                  'MA_total_intensity', 'MA_masked_intensity', 'MA_pixel_count',
#                                                  'MA_total_objects',  # done
#                                                  'MA_masked_intensity_per_pixel', 'MA_masked_intensity_per_object',  # done
#                                                  'MA_total_objects', 'MA_split', 'MA_merge', 'MA_new_id', 'MA_retired_id',
#                                                  'MA_birth', 'MA_death',  # done
#                                                  'MA_split_rate', 'MA_merge_rate', 'MA_birth_rate',
#                                                  'MA_death_rate'])  # done
#
#     file_path = filePath
#     eventsDF = pd.read_csv(file_path + 'all_events.csv')
#
#     time_ = [int(i + 1) for i in range(len(eventsDF))]
#     eventsAndIntensityDF['time'] = time_
#     eventsAndIntensityDF['split'] = eventsDF['Split'].astype('int')
#     eventsAndIntensityDF['merge'] = eventsDF['Merge'].astype('int')
#     eventsAndIntensityDF['new_id'] = eventsDF['new id'].astype('int')
#     eventsAndIntensityDF['retired_id'] = eventsDF['retired id'].astype('int')
#     eventsAndIntensityDF['birth'] = eventsDF['birth'].astype('int')
#     eventsAndIntensityDF['death'] = eventsDF['death'].astype('int')
#     eventsAndIntensityDF['total_objects'] = eventsDF['Total objects'].astype('int')
#
#     mask = niftireadU32(segpath + 'CombinedSO/CombinedSO.nii')[:,:,:,:]
#
#     print('mask shape:  ', mask.shape)
#     newMask = np.zeros_like(mask)
#     newMask[mask > 0] = 1
#     print('new mask shape:  ', newMask.shape)
#
#     image = tifffile.imread(originalImage)
#     image = np.transpose(image, (3, 2, 1, 0))
#     image = image[:, :, :, sT:eT]
#
#     print('image shape:  ', image.shape)
#
#     maskedImage = image * newMask
#     print(np.shape(image))
#
#     distance=endpoint-startpoint+1
#
#     # print(eT, sT, eT-sT-1, startpoint, endpoint, endpoint-startpoint, distance)
#     print('--------------------------------------------------------------------------------------------------------')
#     intensityArr = np.zeros((len(eventsDF), 3))
#     # for i in range(np.size(image, 3)):
#     print("'d,l,i3,mi3,nm3'  -1   :",distance-1,len(eventsDF)-1, image.shape[3]-1, maskedImage.shape[3]-1, newMask.shape[3]-1)
#     for i in range(min(distance-1, len(eventsDF)-1, image.shape[3]-1, maskedImage.shape[3]-1, newMask.shape[3]-1)):
#         intensityArr[i, :] = [image[:, :, :, i].sum(), maskedImage[:, :, :, i].sum(), newMask[:, :, :, i].sum()]
#     intensityDF = pd.DataFrame(intensityArr, columns=['total_intensity', 'masked_intensity', 'pixel_count'])
#     for column in intensityDF.columns:
#         eventsAndIntensityDF[column] = intensityDF[column].astype('int')
#
#     eventsAndIntensityDF['masked_intensity_per_pixel'] = eventsAndIntensityDF['masked_intensity'] / eventsAndIntensityDF['pixel_count']
#     eventsAndIntensityDF['masked_intensity_per_object'] = eventsAndIntensityDF['masked_intensity'] / eventsAndIntensityDF[
#         'total_objects']
#     eventsAndIntensityDF['total_intensity_per_pixel'] = eventsAndIntensityDF['total_intensity'] / eventsAndIntensityDF[
#         'pixel_count']
#     eventsAndIntensityDF['total_intensity_per_object'] = eventsAndIntensityDF['total_intensity'] / eventsAndIntensityDF[
#         'total_objects']
#
#     eventsAndIntensityDF['split_rate'] = eventsAndIntensityDF['split'] / eventsAndIntensityDF['total_objects'].shift(1)
#     eventsAndIntensityDF['merge_rate'] = eventsAndIntensityDF['merge'] / eventsAndIntensityDF['total_objects'].shift(1)
#     eventsAndIntensityDF['birth_rate'] = eventsAndIntensityDF['birth'] / eventsAndIntensityDF['total_objects'].shift(1)
#     eventsAndIntensityDF['death_rate'] = eventsAndIntensityDF['death'] / eventsAndIntensityDF['total_objects'].shift(1)
#
#     for r, m in [('total_objects', 'MA_total_objects'),
#                  ('total_intensity', 'MA_total_intensity'),
#                  ('masked_intensity', 'MA_masked_intensity'),
#                  ('pixel_count', 'MA_pixel_count'),
#                  ('total_objects', 'MA_total_objects'),
#                  ('masked_intensity_per_pixel', 'MA_masked_intensity_per_pixel'),
#                  ('masked_intensity_per_object', 'MA_masked_intensity_per_object'),
#                  ('split', 'MA_split'), ('merge', 'MA_merge'),
#                  ('new_id', 'MA_new_id'), ('retired_id', 'MA_retired_id'),
#                  ('birth', 'MA_birth'), ('death', 'MA_death'),
#                  ('split_rate', 'MA_split_rate'), ('merge_rate', 'MA_merge_rate'),
#                  ('birth_rate', 'MA_birth_rate'),
#                  ('death_rate', 'MA_death_rate')]:
#         eventsAndIntensityDF[m] = eventsAndIntensityDF[r].rolling(window=9, center=True).mean()
#
#     fig, axs = plt.subplots(3, 3, figsize=(40, 20))
#     axs[0, 0].plot(eventsAndIntensityDF['MA_total_intensity'][:-4])
#     axs[0, 0].set_title('Total Intensity')
#     axs[0, 0].legend(labels=['Total Intensity'])
#     axs[0, 1].plot(eventsAndIntensityDF['MA_masked_intensity'][:-4])
#     axs[0, 1].set_title('Masked Intensity')
#     axs[0, 1].legend(labels=['Masked Intensity'])
#     axs[0, 2].plot(eventsAndIntensityDF['MA_total_objects'][:-4])
#     axs[0, 2].set_title('Total Objects')
#     axs[0, 2].legend(labels=['Total Objects'])
#
#     axs[1, 0].plot(eventsAndIntensityDF['MA_masked_intensity_per_object'][:-4])
#     axs[1, 0].set_title('Masked Intensity Per Object')
#     axs[1, 0].legend(labels=['Masked Intensity Per Object'])
#     axs[1, 1].plot(eventsAndIntensityDF['MA_masked_intensity_per_pixel'][:-4])
#     axs[1, 1].set_title('Masked Intensity Per Pixel')
#     axs[1, 1].legend(labels=['Masked Intensity Per Pixel'])
#
#     axs[1, 2].plot(eventsAndIntensityDF['MA_split'][:-4])
#     axs[1, 2].plot(eventsAndIntensityDF['MA_merge'][:-4])
#     axs[1, 2].set_title('Split and Merge')
#     axs[1, 2].legend(labels=['Split', 'Merge'])
#
#     axs[2, 0].plot(eventsAndIntensityDF['MA_birth'][:-4])
#     axs[2, 0].plot(eventsAndIntensityDF['MA_death'][:-4])
#     axs[2, 0].set_title('Birth and Death')
#     axs[2, 0].legend(labels=['Birth', 'Death'])
#
#     axs[2, 1].plot(eventsAndIntensityDF['MA_split_rate'][:-4])
#     axs[2, 1].plot(eventsAndIntensityDF['MA_merge_rate'][:-4])
#     axs[2, 1].set_title('Split Rate and Merge Rate')
#     axs[2, 1].legend(labels=['Split Rate', 'Merge Rate'])
#
#     axs[2, 2].plot(eventsAndIntensityDF['MA_birth_rate'][:-4])
#     axs[2, 2].plot(eventsAndIntensityDF['MA_death_rate'][:-4])
#     axs[2, 2].set_title('Birth Rate and Death Rate')
#     axs[2, 2].legend(labels=['Birth Rate', 'Death Rate'])
#
#     nameOnly = os.path.basename(originalImage)[:-4]
#
#     plt.savefig(file_path + '/' + 'EventsAndIntensityPlots_' + nameOnly + '.png')
#     eventsAndIntensityDF.to_csv(file_path + '/' + 'EventsAndIntensityPlotsData_' + nameOnly + '.csv')
#
#     import time as theTime
#     nowtime = theTime.perf_counter()
#     totalTimeTaken = nowtime-sTime
#     starline()
#     hours_ = int(totalTimeTaken//60//60)
#     min_ = int((totalTimeTaken -  hours_ * 60*60)//60)
#     print(f'Tracking Complete. Total Time: {hours_} hours {min_} min {int(totalTimeTaken%60 + 1)} sec ')
#     starline()


import numpy as np
import pandas as pd
import os
import tifffile
import nibabel as nib
import matplotlib.pyplot as plt
import json

def niftireadUint32(arg):
    return np.asarray(nib.load(arg).dataobj).astype(np.uint32).squeeze()





def runAnalysis(origImgPath, trackedimagepath, sT, eT, plotsavepath):

    def findProgeny(i_d, splitlist, progeny_list, accounted_list):
        if i_d in splitted_:
            progeny_list.extend(splitlist.loc[splitlist['Splitted'] == i_d, 'Splitted Into'].tolist())
            accounted_list.append(i_d)
            for i_d_ in progeny_list:
                if i_d_ not in accounted_list:
                    findProgeny(i_d_, splitlist, progeny_list, accounted_list)
        return progeny_list


    image = tifffile.imread(origImgPath)
    acp=20
    image = np.transpose(image, (3, 2, 1, 0))
    image = image[:, :, :, sT - 1:eT]
    countsavename = os.path.join(os.path.dirname(trackedimagepath), 'pixelCountPerObjectEveryTimepoint.csv')
    matrix = niftireadUint32(trackedimagepath)
    matrix_shape = matrix.shape
    mask___ = np.zeros_like(matrix)
    mask___[matrix > 0] = 1
    masked_image = image * mask___

    totalIntensity = {}
    maskedTotalIntensity = {}

    for _t_ in range(matrix.shape[-1]):
        totalIntensity[_t_ + 1] = np.sum(image[:, :, :, _t_])
        maskedTotalIntensity[_t_ + 1] = np.sum(masked_image[:, :, :, _t_])

    total_intensity_df_ = pd.DataFrame(list(totalIntensity.values()))
    masked_intensity_df_ = pd.DataFrame(list(maskedTotalIntensity.values()))
    total_intensity_df_.columns = ['totalIntensity']
    masked_intensity_df_.columns = ['TotalIntensity(masked)']

    flattened_matrix1 = matrix.flatten()
    unique_values1 = np.unique(flattened_matrix1, return_counts=False)
    counts_df = pd.DataFrame(index=np.arange(1, max(unique_values1)),
                             columns=np.arange(1, matrix.shape[-1] + 1)).fillna(0).astype(int)
    unique_values1 = unique_values1[1:]
    #     countsall = countsall[1:] # ignore zero counts
    countsall = []
    for _t_ in range(matrix.shape[-1]):
        v, c = np.unique(matrix[:, :, :, _t_], return_counts=True)
        v = v[1:];
        c = c[1:]
        countsall = countsall + list(c)  # append the content of c
    q1, m, q2, q3 = int(np.percentile(countsall, 25)), int(np.median(countsall)), int(
        np.percentile(countsall, 50)), int(np.percentile(countsall, 75))
    print(' q1= ', q1, ' q2= ', q2, ' q3= ', q3, ' acp = ', acp)

    ################################################################################################

    int_q1 = np.zeros_like(matrix)
    int_q2 = np.zeros_like(matrix)
    int_q3 = np.zeros_like(matrix)
    int_q4 = np.zeros_like(matrix)

    maskedIntensityBySize = {}

    for _t_ in range(matrix.shape[-1]):
        mat_t_ = matrix[:, :, :, _t_]
        _u_, _c_ = np.unique(mat_t_, return_counts=True)
        _u_ = _u_[1:]
        _c_ = _c_[1:]
        _u_q1 = _u_[np.where(_c_ <= q1)]
        _u_q2 = _u_[np.where((_c_ <= q2) & (_c_ > q1))]
        _u_q3 = _u_[np.where((_c_ <= q3) & (_c_ > q2))]
        _u_q4 = _u_[np.where(_c_ > q3)]

        maskq1 = np.isin(mat_t_, _u_q1)
        maskq2 = np.isin(mat_t_, _u_q2)
        maskq3 = np.isin(mat_t_, _u_q3)
        maskq4 = np.isin(mat_t_, _u_q4)

        int_q1[:, :, :, _t_] = np.where(maskq1, image[:, :, :, _t_], 0)
        int_q2[:, :, :, _t_] = np.where(maskq2, image[:, :, :, _t_], 0)
        int_q3[:, :, :, _t_] = np.where(maskq3, image[:, :, :, _t_], 0)
        int_q4[:, :, :, _t_] = np.where(maskq4, image[:, :, :, _t_], 0)

        maskedIntensityBySize[_t_ + 1] = [np.mean(int_q1[:, :, :, _t_]), np.mean(int_q2[:, :, :, _t_]),
                                          np.mean(int_q3[:, :, :, _t_]), np.mean(int_q4[:, :, :, _t_])]

    maskedIntensityBySize_df = pd.DataFrame(maskedIntensityBySize.values(), columns=['leq1', 'leq2', 'leq3', 'gq3'])

    ################################################################################################
    #     ################################################################################################

    print('birth/death/total')

    all_t_by_size = {}

    for t in range(matrix.shape[-1]):
        matrix_at_t_ = matrix[:, :, :, t]

        all_ids_at_t_, all_ids_at_t_count = np.unique(matrix_at_t_, return_counts=True)

        all_ids_at_t_ = all_ids_at_t_[1:]
        all_ids_at_t_count = all_ids_at_t_count[1:]

        all_lq1_t = sum(1 for element in all_ids_at_t_count if element <= q1)
        all_lq2_t = sum(1 for element in all_ids_at_t_count if element > q1 and element <= q2)
        all_lq3_t = sum(1 for element in all_ids_at_t_count if element > q2 and element <= q3)
        all_gq3_t = sum(1 for element in all_ids_at_t_count if element > q3)

        all_t_by_size[t] = [all_lq1_t, all_lq2_t, all_lq3_t, all_gq3_t]  # count of all objects by size

    bornlist = {}
    deadlist = {}

    # bornlist = json.loads(os.path.join(plotsavepath, 'born_id.json'))
    # deadlist = json.loads(os.path.join(plotsavepath, 'dead_id.json'))

    split_csv_ = pd.read_csv(os.path.join(os.path.dirname(trackedimagepath), 'split_list.csv'))
    merge_csv_ = pd.read_csv(os.path.join(os.path.dirname(trackedimagepath), 'merge_list.csv'))

    for t in range(matrix.shape[-1] - 2):
        # all split ids at timepoint t+1
        split_id_list_at_t_ = split_csv_.loc[split_csv_['Time'] == t + 2, 'Splitted Into'].tolist()
        merged_id_list_at_t_ = merge_csv_.loc[merge_csv_['Time'] == t + 1, 'Merged'].tolist()
        unique_t = np.unique(matrix[:, :, :, t])[1:]
        unique_t1 = np.unique(matrix[:, :, :, t + 1])[1:]
        unique_t2 = np.unique(matrix[:, :, :, t + 2])[1:]
        bornlist[t + 1] = [it for it in [item for item in unique_t1 if item not in unique_t] if
                           it not in split_id_list_at_t_]
        deadlist[t] = [it for it in [item for item in unique_t if item not in unique_t1] if
                       it not in merged_id_list_at_t_]

    born_by_size = {}
    born_by_size_frac = {}
    dead_by_size = {}
    dead_by_size_frac = {}

    for bk in bornlist.keys():
        mask_bt = np.isin(matrix[:, :, :, bk], bornlist[bk])
        matrix_btm = matrix[:, :, :, bk][mask_bt]
        id_b, count_b = np.unique(matrix_btm, return_counts=True)

        lq1_b = sum(1 for element in count_b if element <= q1)
        lq2_b = sum(1 for element in count_b if element > q1 and element <= q2)
        lq3_b = sum(1 for element in count_b if element > q2 and element <= q3)
        gq3_b = sum(1 for element in count_b if element > q3)

        born_by_size[bk] = [lq1_b, lq2_b, lq3_b, gq3_b]

    #         born_by_size_frac[bk+1] = [x/y if y!= 0 else 0 for x,y in zip(born_by_size[bk], all_t_by_size[bk])]

    for dk in deadlist.keys():
        mask_dt = np.isin(matrix[:, :, :, dk], deadlist[dk])
        matrix_dtm = matrix[:, :, :, dk][mask_dt]
        id_d, count_d = np.unique(matrix_dtm, return_counts=True)

        lq1_d = sum(1 for element in count_d if element <= q1)
        lq2_d = sum(1 for element in count_d if element > q1 and element <= q2)
        lq3_d = sum(1 for element in count_d if element > q2 and element <= q3)
        gq3_d = sum(1 for element in count_d if element > q3)

        dead_by_size[dk] = [lq1_d, lq2_d, lq3_d, gq3_d]

        dead_by_size_frac[dk + 1] = [x / y if y != 0 else 0 for x, y in zip(dead_by_size[dk], all_t_by_size[dk])]
    #     print(born_by_size, dead_by_size)
    #     stop
    born_by_size_df = pd.DataFrame(born_by_size.values())
    born_by_size_df.columns = ['b size <= q1', 'b q1 < size <= q2', 'b q2 < size <= q3', 'b size > q3']
    dead_by_size_df = pd.DataFrame(dead_by_size_frac.values())
    dead_by_size_df.columns = ['d size <= q1', 'd q1 < size <= q2', 'd q2 < size <= q3', 'd size > q3']

    print('      birth/death/total')

    all_t_by_size_df = pd.DataFrame(list(all_t_by_size.values()))
    all_t_by_size_df.columns = ['all size <= q1', 'all q1 < size <= q2', 'all q2 < size <= q3', 'all size > q3']
    #     print(np.sum(np.array(list(all_t_by_size.values())), axis=0))

    #     ################################################################################################

    ############################################################################################################################################################
    # Split by size over time
    lq1_count_split = 0
    lq2_count_split = 0
    lq3_count_split = 0
    gq3_count_split = 0

    all_timepoints_by_size = {}
    split_by_size = {}
    split_by_size_frac = {}
    merged_into_by_size = {}
    merged_by_size = {}
    merged_into_by_size_frac = {}
    merged_by_size_frac = {}

    split_csv_ = pd.read_csv(os.path.join(os.path.dirname(trackedimagepath), 'split_list.csv'))
    merge_csv_ = pd.read_csv(os.path.join(os.path.dirname(trackedimagepath), 'merge_list.csv'))

    unique_split_t_ = np.unique(split_csv_['Time'])
    unique_merge_t_ = np.unique(merge_csv_['Time'])

    print('split')

    for ust in unique_split_t_:
        matrix_at_ust_ = matrix[:, :, :, ust - 2]

        all_ids_at_t_, all_ids_at_t_count = np.unique(matrix_at_ust_, return_counts=True)

        all_ids_at_t_ = all_ids_at_t_[1:]
        all_ids_at_t_count = all_ids_at_t_count[1:]

        all_lq1_t = sum(1 for element in all_ids_at_t_count if element <= q1)
        all_lq2_t = sum(1 for element in all_ids_at_t_count if element > q1 and element <= q2)
        all_lq3_t = sum(1 for element in all_ids_at_t_count if element > q2 and element <= q3)
        all_gq3_t = sum(1 for element in all_ids_at_t_count if element > q3)

        all_timepoints_by_size[ust - 1] = [all_lq1_t, all_lq2_t, all_lq3_t, all_gq3_t]  # count of all objects by size

        split_id_list_at_t_ = split_csv_.loc[split_csv_['Time'] == ust, 'Splitted'].tolist()

        mask_ = np.isin(matrix_at_ust_, split_id_list_at_t_)

        matrix_at_ust_ = matrix_at_ust_[mask_]

        id_, count_ = np.unique(matrix_at_ust_, return_counts=True)

        lq1_count_split = sum(1 for element in count_ if element <= q1)
        lq2_count_split = sum(1 for element in count_ if element > q1 and element <= q2)
        lq3_count_split = sum(1 for element in count_ if element > q2 and element <= q3)
        gq3_count_split = sum(1 for element in count_ if element > q3)

        split_by_size[ust - 1] = [lq1_count_split, lq2_count_split, lq3_count_split, gq3_count_split]

        split_by_size_frac[ust - 1] = [
            lq1_count_split / all_lq1_t if all_lq1_t != 0 else 0,
            lq2_count_split / all_lq2_t if all_lq2_t != 0 else 0,
            lq3_count_split / all_lq3_t if all_lq3_t != 0 else 0,
            gq3_count_split / all_gq3_t if all_gq3_t != 0 else 0
        ]

    split_by_size_df = pd.DataFrame(split_by_size_frac).T
    split_by_size_df.columns = ['s size <= q1', 's q1 < size <= q2', 's q2 < size <= q3', 's size > q3']

    print('      split')

    print('merge')
    for ust in unique_merge_t_:
        #         lq1_mi = 0; lq2_mi = 0; lq3_mi = 0; gq3_mi = 0
        #         lq1_m = 0; lq2_m = 0; lq3_m = 0; gq3_m = 0

        matrix_at_ust_ = matrix[:, :, :, ust - 2]

        all_ids_at_t_, all_ids_at_t_count = np.unique(matrix_at_ust_, return_counts=True)

        all_ids_at_t_ = all_ids_at_t_[1:]
        all_ids_at_t_count = all_ids_at_t_count[1:]

        all_lq1_t = sum(1 for element in all_ids_at_t_count if element <= q1)
        all_lq2_t = sum(1 for element in all_ids_at_t_count if element > q1 and element <= q2)
        all_lq3_t = sum(1 for element in all_ids_at_t_count if element > q2 and element <= q3)
        all_gq3_t = sum(1 for element in all_ids_at_t_count if element > q3)

        all_timepoints_by_size[ust - 1] = [all_lq1_t, all_lq2_t, all_lq3_t, all_gq3_t]  # count of all objects by size

        merged_into_id_list_at_t_ = merge_csv_.loc[merge_csv_['Time'] == ust, 'Merged Into'].tolist()
        merged_id_list_at_t = merge_csv_.loc[merge_csv_['Time'] == ust, 'Merged'].tolist()

        count_mi = [np.bincount(matrix_at_ust_.flatten())[element] for element in merged_into_id_list_at_t_]
        count_m = [np.bincount(matrix_at_ust_.flatten())[element] for element in merged_id_list_at_t]
        #         print(count_mi, count_m)

        lq1_mi = sum(1 for element in count_mi if element <= q1)
        lq2_mi = sum(1 for element in count_mi if element > q1 and element <= q2)
        lq3_mi = sum(1 for element in count_mi if element > q2 and element <= q3)
        gq3_mi = sum(1 for element in count_mi if element > q3)

        lq1_m = sum(1 for element in count_m if element <= q1)
        lq2_m = sum(1 for element in count_m if element > q1 and element <= q2)
        lq3_m = sum(1 for element in count_m if element > q2 and element <= q3)
        gq3_m = sum(1 for element in count_m if element > q3)

        merged_into_by_size[ust - 1] = [lq1_mi, lq2_mi, lq3_mi, gq3_mi]
        merged_by_size[ust - 1] = [lq1_m, lq2_m, lq3_m, gq3_m]

        merged_into_by_size_frac[ust - 1] = [
            lq1_mi / all_lq1_t if all_lq1_t != 0 else 0,
            lq2_mi / all_lq2_t if all_lq2_t != 0 else 0,
            lq3_mi / all_lq3_t if all_lq3_t != 0 else 0,
            gq3_mi / all_gq3_t if all_gq3_t != 0 else 0
        ]

        merged_by_size_frac[ust - 1] = [
            lq1_m / all_lq1_t if all_lq1_t != 0 else 0,
            lq2_m / all_lq2_t if all_lq2_t != 0 else 0,
            lq3_m / all_lq3_t if all_lq3_t != 0 else 0,
            gq3_m / all_gq3_t if all_gq3_t != 0 else 0
        ]

    merged_into_by_size_df = pd.DataFrame(merged_into_by_size_frac).T
    merged_by_size_df = pd.DataFrame(merged_by_size_frac).T

    merged_into_by_size_df.columns = ['mp size <= q1', 'mp q1 < size <= q2', 'mp q2 < size <= q3', 'mp size > q3']
    merged_by_size_df.columns = ['ms size <= q1', 'ms q1 < size <= q2', 'ms q2 < size <= q3', 'ms size > q3']

    print('      merge')
    #     merged_by_size_df.plot()

    ##################################################################################################################
    ##################################################################################################################
    split_csv_ = pd.read_csv(os.path.join(os.path.dirname(trackedimagepath), 'split_list.csv'))
    merge_csv_ = pd.read_csv(os.path.join(os.path.dirname(trackedimagepath), 'merge_list.csv'))

    bornlist = {}
    for t in range(matrix.shape[-1] - 1):
        # all split ids at timepoint t+1
        split_id_list_at_t_ = split_csv_.loc[split_csv_['Time'] == t + 2, 'Splitted Into'].tolist()
        unique_t = np.unique(matrix[:, :, :, t])[1:]
        unique_t1 = np.unique(matrix[:, :, :, t + 1])[1:]
        bornlist[t] = [it for it in [item for item in unique_t1 if item not in unique_t] if
                       it not in split_id_list_at_t_]

    lq1, bq1q2, bq2q3, gq3 = {}, {}, {}, {}
    print('expectancy')

    for t in range(matrix.shape[3] - 1):
        flattened_matrix = matrix[:, :, :, t].flatten()
        unique_values, value_counts = np.unique(flattened_matrix, return_counts=True)
        unique_values = unique_values[1:]
        value_counts = value_counts[1:]

        lq1t, bq1q2t, bq2q3t, gq3t = [], [], [], []

        for index_, i_vc in enumerate(unique_values):
            if value_counts[index_] <= q1:
                lq1t.append(unique_values[index_])
            elif value_counts[index_] <= q2:
                bq1q2t.append(unique_values[index_])
            elif value_counts[index_] <= q3:
                bq2q3t.append(unique_values[index_])
            else:
                gq3t.append(unique_values[index_])
        lq1[t] = lq1t
        bq1q2[t] = bq1q2t
        bq2q3[t] = bq2q3t
        gq3[t] = gq3t

        for index__, uv in enumerate(unique_values):
            counts_df.loc[uv, t] = value_counts[index__]

    # Save the counts as a CSV file
    counts_df.to_csv(countsavename, index=False)

    ##################################

    splitlist = pd.read_csv(os.path.join(os.path.dirname(trackedimagepath), 'split_list.csv'))
    splitted_ = splitlist['Splitted']
    splitted_into_ = splitlist['Splitted Into']

    all_id_and_progeny = {}

    for i_d in unique_values1:
        progeny_list = findProgeny(i_d, splitlist, [i_d], [])
        all_id_and_progeny[i_d] = progeny_list

    #     print(all_id_and_progeny)
    ##################################
    uniquexxx = {}
    for t in range(matrix.shape[3]):
        uniquexxx[t] = np.unique(matrix[:, :, :, t].flatten())

    last_t_of_all_ids_and_their_progeny = {}
    last_t_accounted_list = []

    for t in range(matrix.shape[3] - 1, -1, -1):
        current_list = []
        for i_d in all_id_and_progeny.keys():
            if set(all_id_and_progeny[i_d]).intersection(uniquexxx[t]) and i_d not in last_t_accounted_list:
                current_list.append(i_d)
                last_t_of_all_ids_and_their_progeny[i_d] = t
                last_t_accounted_list.append(i_d)

    last_t_of_all_ids_and_their_progeny_keys = list(last_t_of_all_ids_and_their_progeny.keys())
    last_t_of_all_ids_and_their_progeny_keys.sort()
    last_t_of_all_ids_and_their_progeny = {i: last_t_of_all_ids_and_their_progeny[i] for i in
                                           last_t_of_all_ids_and_their_progeny_keys}

    ##################################
    life_exp_lq1 = {}
    life_exp_bq1q2 = {}
    life_exp_bq2q3 = {}
    life_exp_gq3 = {}

    for t in range(matrix.shape[3] - 1):
        life_exp_lq1[t] = np.mean([last_t_of_all_ids_and_their_progeny[id_] - t + 1 for id_ in lq1[t]])
        life_exp_bq1q2[t] = np.mean([last_t_of_all_ids_and_their_progeny[id_] - t + 1 for id_ in bq1q2[t]])
        life_exp_bq2q3[t] = np.mean([last_t_of_all_ids_and_their_progeny[id_] - t + 1 for id_ in bq2q3[t]])
        life_exp_gq3[t] = np.mean([last_t_of_all_ids_and_their_progeny[id_] - t + 1 for id_ in gq3[t]])

        ##
    life_exp_lq1_new = {};
    life_exp_lq2_new = {};
    life_exp_lq3_new = {};
    life_exp_gq3_new = {}
    lq1_new = {};
    lq2_new = {};
    lq3_new = {};
    gq3_new = {}
    for k_ey in bornlist.keys():
        if k_ey < max(bornlist.keys()):
            lq1_new[k_ey + 1] = [value for value in lq1[k_ey + 1] if value in bornlist[k_ey]]
            lq2_new[k_ey + 1] = [value for value in bq1q2[k_ey + 1] if value in bornlist[k_ey]]
            lq3_new[k_ey + 1] = [value for value in bq2q3[k_ey + 1] if value in bornlist[k_ey]]
            gq3_new[k_ey + 1] = [value for value in gq3[k_ey + 1] if value in bornlist[k_ey]]
    allbornlist = [value for value_list in bornlist.values() for value in value_list]
    last_t_of_all_ids_and_their_progeny_new = {i: last_t_of_all_ids_and_their_progeny[i] for i in
                                               last_t_of_all_ids_and_their_progeny_keys if i in allbornlist}

    for t in range(1, matrix.shape[3] - 1):
        life_exp_lq1_new[t] = np.mean([last_t_of_all_ids_and_their_progeny_new[id_] - t + 1 for id_ in lq1_new[t]])
        life_exp_lq2_new[t] = np.mean([last_t_of_all_ids_and_their_progeny_new[id_] - t + 1 for id_ in lq2_new[t]])
        life_exp_lq3_new[t] = np.mean([last_t_of_all_ids_and_their_progeny_new[id_] - t + 1 for id_ in lq3_new[t]])
        life_exp_gq3_new[t] = np.mean([last_t_of_all_ids_and_their_progeny_new[id_] - t + 1 for id_ in gq3_new[t]])

    life_exp_new = np.array([list(s) for (s) in list(zip(life_exp_lq1_new.values(),
                                                         life_exp_lq2_new.values(),
                                                         life_exp_lq3_new.values(),
                                                         life_exp_gq3_new.values()
                                                         )
                                                     )
                             ]
                            )

    life_exp_new_df = pd.DataFrame(life_exp_new,
                                   columns=['en size less than or equal to q1', 'en size between q1 and q2',
                                            'en size between q2 and q3', 'en size greater than q3'])

    ##

    life_expectancies = np.array([list(s) for (s) in list(zip(life_exp_lq1.values(),
                                                              life_exp_bq1q2.values(),
                                                              life_exp_bq2q3.values(),
                                                              life_exp_gq3.values()))])

    life_expectancies_df = pd.DataFrame(life_expectancies,
                                        columns=['e size less than or equal to q1', 'e size between q1 and q2',
                                                 'e size between q2 and q3', 'e size greater than q3'])
    life_expectancies_df.to_csv(os.path.join(os.path.dirname(trackedimagepath), 'Life_expectancy_by_size.csv'),
                                index=False)
    # life_expectancies_df

    print('      expectancy')

    ######################################################################################################################
    #####################################################################################################################################################################

    print('csv and plots')
    combined_df = pd.concat([split_by_size_df, merged_into_by_size_df, merged_by_size_df,
                             born_by_size_df, dead_by_size_df, all_t_by_size_df, maskedIntensityBySize_df,
                             masked_intensity_df_,
                             life_expectancies_df, life_exp_new_df], axis=1)

    combined_df.to_csv(os.path.join(plotsavepath, 'Size_Distribution_of_Events_and_Expectancy_over_time_' +
                                    os.path.basename(os.path.dirname(trackedimagepath)) + '_acp_int.csv'))

    fig, axes = plt.subplots(4, 5, figsize=(50, 40))

    dfs = [split_by_size_df, merged_into_by_size_df, merged_by_size_df, born_by_size_df, dead_by_size_df,
           all_t_by_size_df,  # total_intensity_df_,
           maskedIntensityBySize_df,
           masked_intensity_df_, life_expectancies_df, life_exp_new_df]

    titles = ['Fraction of Objects of Each Quartile Undergoing Split Event',
              'Fraction of Objects of Each Quartile Undergoing Merge Event as "Primary"',
              'Fraction of Objects of Each Quartile Undergoing Merge Event as "Secondary"',
              'Number of Objects Born into a Quartile',
              'Fraction of Objects Dead from a Quartile',
              'Total Objects in Each Quartile',  # 'Total Image Intensity',
              'Masked Intensity By Size',
              'Object Intensity (masked total intensity)', 'Avg. Life Expectancy By Size',
              'Avg. Life Expectancy of New Objects By Size'
              ]

    y_axs = ['Fraction of Objects of the Same Group',
             'Fraction of Objects of the Same Group',
             'Fraction of Objects of the Same Group',
             'Number of Objects of the Same Group',
             'Fraction of Objects of the Same Group',
             'Number of Objects',  # 'Intensity Value',
             'Intensity of Each Group',
             'Intensity Value', 'Life Expectancy at Timepoints', 'Life Expectancy of New Obj.'
             ]

    colors = ['blue', 'purple', 'magenta', 'red']

    for _ in range(4):
        for __ in range(int(len(dfs) // 2 + len(dfs) % 2)):

            if _ < 2:
                ax = axes[_, __]
                if _ < 1 and __ < 3:
                    ax.set_ylim(0, 0.5)
                df = dfs[_ * 5 + __]
                # df.rolling(window=9, min_periods=1).mean().plot(ax=ax)
                df.plot(ax=ax, color=colors)
                ax.axvline(x=acp, color='green', linestyle='--', label='apical constriction point')
                ax.set_title(titles[_ * 5 + __], fontsize=15)
                ax.set_xlabel('timepoint', fontsize=15)
                ax.set_ylabel(y_axs[_ * 5 + __], fontsize=15)
                ax.legend(fontsize=15)
            else:
                ax = axes[_, __]
                if _ < 3 and __ < 3:
                    ax.set_ylim(0, 0.5)
                df = dfs[(_ - 2) * 5 + __]
                df.rolling(window=21, min_periods=1).mean().plot(ax=ax, color=colors)
                ax.axvline(x=acp, color='green', linestyle='--', label='apical constriction point')
                ax.set_title(titles[(_ - 2) * 5 + __] + ' (Moving Average)', fontsize=15)
                ax.set_xlabel('timepoint', fontsize=15)
                ax.set_ylabel(y_axs[(_ - 2) * 5 + __], fontsize=15)
                ax.legend(fontsize=15)
    plt.tight_layout()

    plt.savefig(os.path.join(plotsavepath, 'Size_Distribution_of_Events_and_Expectancy_over_time_' + os.path.basename(
        os.path.dirname(trackedimagepath)) + '.png'), facecolor='white')
    plt.close()
    print('      csv and plots')
    print('-------------------------------')

#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################

import numpy as np
import pandas as pd
import os
import tifffile
import nibabel as nib
import matplotlib.pyplot as plt
import json

def niftireadUint32(arg):
    return np.asarray(nib.load(arg).dataobj).astype(np.uint32).squeeze()

def runAnalysisNewWay(origImgPath, trackedimagepath, sT, eT, plotsavepath):

    def findProgeny(i_d, splitlist, progeny_list, accounted_list):
        if i_d in splitted_:
            progeny_list.extend(splitlist.loc[splitlist['Splitted'] == i_d, 'Splitted Into'].tolist())
            accounted_list.append(i_d)
            for i_d_ in progeny_list:
                if i_d_ not in accounted_list:
                    findProgeny(i_d_, splitlist, progeny_list, accounted_list)
        return progeny_list

    image = tifffile.imread(origImgPath)
    # acp = 20
    image = np.transpose(image, (3, 2, 1, 0))
    image = image[:, :, :, sT - 1:eT]
    countsavename = os.path.join(os.path.dirname(trackedimagepath), 'pixelCountPerObjectEveryTimepoint.csv')
    matrix = niftireadUint32(trackedimagepath)
    matrix_shape = matrix.shape
    mask___ = np.zeros_like(matrix)
    mask___[matrix > 0] = 1
    masked_image = image * mask___

    totalIntensity = {}
    maskedTotalIntensity = {}

    for _t_ in range(matrix.shape[-1]):
        totalIntensity[_t_ + 1] = np.sum(image[:, :, :, _t_])
        maskedTotalIntensity[_t_ + 1] = np.sum(masked_image[:, :, :, _t_])

    total_intensity_df_ = pd.DataFrame(list(totalIntensity.values()))
    masked_intensity_df_ = pd.DataFrame(list(maskedTotalIntensity.values()))
    total_intensity_df_.columns = ['totalIntensity']
    masked_intensity_df_.columns = ['TotalIntensity(masked)']

    flattened_matrix1 = matrix.flatten()
    unique_values1 = np.unique(flattened_matrix1, return_counts=False)
    counts_df = pd.DataFrame(index=np.arange(1, max(unique_values1)),
                             columns=np.arange(1, matrix.shape[-1] + 1)).fillna(0).astype(int)
    unique_values1 = unique_values1[1:]
    #     countsall = countsall[1:] # ignore zero counts
    countsall = []
    for _t_ in range(matrix.shape[-1]):
        v, c = np.unique(matrix[:, :, :, _t_], return_counts=True)
        v = v[1:];
        c = c[1:]
        countsall = countsall + list(c)  # append the content of c
    q1, m, q2, q3 = int(np.percentile(countsall, 25)), int(np.median(countsall)), int(
        np.percentile(countsall, 50)), int(np.percentile(countsall, 75))
    # print(' q1= ', q1, ' q2= ', q2, ' q3= ', q3, ' acp = ', acp)

    ################################################################################################

    int_q1 = np.zeros_like(matrix)
    int_q2 = np.zeros_like(matrix)
    int_q3 = np.zeros_like(matrix)
    int_q4 = np.zeros_like(matrix)

    maskedIntensityBySize = {}

    for _t_ in range(matrix.shape[-1]):
        mat_t_ = matrix[:, :, :, _t_]
        _u_, _c_ = np.unique(mat_t_, return_counts=True)
        _u_ = _u_[1:]
        _c_ = _c_[1:]
        _u_q1 = _u_[np.where(_c_ <= q1)]
        _u_q2 = _u_[np.where((_c_ <= q2) & (_c_ > q1))]
        _u_q3 = _u_[np.where((_c_ <= q3) & (_c_ > q2))]
        _u_q4 = _u_[np.where(_c_ > q3)]

        maskq1 = np.isin(mat_t_, _u_q1)
        maskq2 = np.isin(mat_t_, _u_q2)
        maskq3 = np.isin(mat_t_, _u_q3)
        maskq4 = np.isin(mat_t_, _u_q4)

        int_q1[:, :, :, _t_] = np.where(maskq1, image[:, :, :, _t_], 0)
        int_q2[:, :, :, _t_] = np.where(maskq2, image[:, :, :, _t_], 0)
        int_q3[:, :, :, _t_] = np.where(maskq3, image[:, :, :, _t_], 0)
        int_q4[:, :, :, _t_] = np.where(maskq4, image[:, :, :, _t_], 0)

        maskedIntensityBySize[_t_ + 1] = [np.mean(int_q1[:, :, :, _t_]), np.mean(int_q2[:, :, :, _t_]),
                                          np.mean(int_q3[:, :, :, _t_]), np.mean(int_q4[:, :, :, _t_])]

    maskedIntensityBySize_df = pd.DataFrame(maskedIntensityBySize.values(), columns=['leq1', 'leq2', 'leq3', 'gq3'])

    ################################################################################################
    print('birth/death/total')
    all_t_by_size = {}
    bornlist = {}
    deadlist = {}

    all_ids = {}
    with open(os.path.join(plotsavepath,'all_id.json'), 'r') as f:
        all_ids = json.load(f)
    all_ids = {int(key):value for key,value in all_ids.items()}

    for ak in all_ids.keys():
        mask_all = np.isin(matrix[:, :, :, ak - 1], all_ids[ak])
        matrix_at_t_ = matrix[:, :, :, ak-1]
        matrix_all = matrix[:, :, :, ak - 1][mask_all]
        id_all, count_all = np.unique(matrix_all, return_counts=True)

        lq1_all = sum(1 for element in count_all if element <= q1)
        lq2_all = sum(1 for element in count_all if element > q1 and element <= q2)
        lq3_all = sum(1 for element in count_all if element > q2 and element <= q3)
        gq3_all = sum(1 for element in count_all if element > q3)

        all_t_by_size[ak] = [lq1_all, lq2_all, lq3_all, gq3_all]


    for t in range(matrix.shape[-1]):
        matrix_at_t_ = matrix[:, :, :, t]

        all_ids_at_t_, all_ids_at_t_count = np.unique(matrix_at_t_, return_counts=True)

        all_ids_at_t_ = all_ids_at_t_[1:]
        all_ids_at_t_count = all_ids_at_t_count[1:]

        all_lq1_t = sum(1 for element in all_ids_at_t_count if element <= q1)
        all_lq2_t = sum(1 for element in all_ids_at_t_count if element > q1 and element <= q2)
        all_lq3_t = sum(1 for element in all_ids_at_t_count if element > q2 and element <= q3)
        all_gq3_t = sum(1 for element in all_ids_at_t_count if element > q3)

        all_t_by_size[t+1] = [all_lq1_t, all_lq2_t, all_lq3_t, all_gq3_t]  # count of all objects by size



    with open(os.path.join(plotsavepath,'born_id.json'), 'r') as f:
        bornlist = json.load(f)
    bornlist = {int(key):value for key,value in bornlist.items()}
    with open(os.path.join(plotsavepath,'dead_id.json'), 'r') as f:
        deadlist = json.load(f)
    deadlist = {int(key):value for key,value in deadlist.items()}

    born_by_size = {}
    dead_by_size = {}
    dead_by_size_frac = {}

    for bk in bornlist.keys():
        if bk!=1: # All objects in  the first frame are kept in bornlist, but don't reflect the true essense of object 'birth'
            mask_bt = np.isin(matrix[:, :, :, bk-1], bornlist[bk])
            matrix_btm = matrix[:, :, :, bk-1][mask_bt]
            id_b, count_b = np.unique(matrix_btm, return_counts=True)

            lq1_b = sum(1 for element in count_b if element <= q1)
            lq2_b = sum(1 for element in count_b if element > q1 and element <= q2)
            lq3_b = sum(1 for element in count_b if element > q2 and element <= q3)
            gq3_b = sum(1 for element in count_b if element > q3)

            born_by_size[bk] = [lq1_b, lq2_b, lq3_b, gq3_b]

    for dk in deadlist.keys():
        mask_dt = np.isin(matrix[:, :, :, dk-2], deadlist[dk])
        matrix_dtm = matrix[:, :, :, dk-2][mask_dt]
        id_d, count_d = np.unique(matrix_dtm, return_counts=True)

        lq1_d = sum(1 for element in count_d if element <= q1)
        lq2_d = sum(1 for element in count_d if element > q1 and element <= q2)
        lq3_d = sum(1 for element in count_d if element > q2 and element <= q3)
        gq3_d = sum(1 for element in count_d if element > q3)

        dead_by_size[dk-1] = [lq1_d, lq2_d, lq3_d, gq3_d] # adjusted to dk-1 so that the deaths encountered for dk are of objects only present until dk-1

        dead_by_size_frac[dk - 1] = [x / y if y != 0 else 0 for x, y in zip(dead_by_size[dk-1], all_t_by_size[dk-1])] # Fraction of objects in dk-1 dead

    born_by_size_df = pd.DataFrame(born_by_size.values())
    born_by_size_df.columns = ['birth size <= q1', 'q1 < size <= q2', 'q2 < size <= q3', 'size > q3']
    dead_by_size_df = pd.DataFrame(dead_by_size_frac.values())
    dead_by_size_df.columns = ['death size <= q1', 'd q1 < size <= q2', 'd q2 < size <= q3', 'd size > q3']
    all_t_by_size_df = pd.DataFrame(list(all_t_by_size.values()))
    all_t_by_size_df.columns = ['all size <= q1', 'all q1 < size <= q2', 'all q2 < size <= q3', 'all size > q3']

    print('      birth/death/total')

    print('split')

    splitting_ids = {}
    split_by_size = {}
    split_by_size_frac = {}

    with open(os.path.join(plotsavepath,'splitting_id.json'), 'r') as f:
        splitting_ids = json.load(f)
    splitting_ids = {int(key):value for key,value in splitting_ids.items()}

    for sk in splitting_ids.keys():
        mask_st = np.isin(matrix[:, :, :, sk-2], splitting_ids[sk])
        matrix_stm = matrix[:, :, :, sk-2][mask_st]
        id_s, count_s = np.unique(matrix_stm, return_counts=True)

        lq1_s = sum(1 for element in count_s if element <= q1)
        lq2_s = sum(1 for element in count_s if element > q1 and element <= q2)
        lq3_s = sum(1 for element in count_s if element > q2 and element <= q3)
        gq3_s = sum(1 for element in count_s if element > q3)

        split_by_size[sk-1] = [lq1_s, lq2_s, lq3_s, gq3_s] # adjusted to sk-1 so that the splits encountered for dk are of objects present at dk-1

        split_by_size_frac[sk - 1] = [x / y if y != 0 else 0 for x, y in zip(split_by_size[sk-1], all_t_by_size[sk-1])] # Fraction of objects in dk-1 dead
    split_by_size_df = pd.DataFrame(list(split_by_size_frac.values()))
    split_by_size_df.columns = ['split size <= q1', 'q1 < size <= q2', 'q2 < size <= q3', 'size > q3']

    print('      split')

    print('merge')

    merge_primary_ids = {}
    merge_secondary_ids = {}
    mergePrimarySize = {}
    mergeSecondarySize = {}
    mergePrimarySizeFrac = {}
    mergeSecondarySizeFrac = {}


    with open(os.path.join(plotsavepath,'truemerge_as_primary.json'),'r') as f:
        merge_primary_ids = json.load(f)
    merge_primary_ids = {int(key):value for key,value in merge_primary_ids.items()}

    with open(os.path.join(plotsavepath, 'truemerge_as_secondary.json'),'r') as f:
        merge_secondary_ids  = json.load(f)
    merge_secondary_ids = {int(key):value for key,value in merge_secondary_ids.items()}


    for msk in merge_secondary_ids.keys():
        mask_mst = np.isin(matrix[:,:,:,msk-2],merge_secondary_ids[msk])
        matrix_mstm = matrix[:,:,:,msk-2][mask_mst]
        id_ms, count_ms = np.unique(matrix_mstm, return_counts=True)

        lq1_ms = sum(1 for element in count_ms if element <= q1)
        lq2_ms = sum(1 for element in count_ms if element > q1 and element <= q2)
        lq3_ms = sum(1 for element in count_ms if element > q2 and element <= q3)
        gq3_ms = sum(1 for element in count_ms if element > q3)

        mergeSecondarySize[msk-1] = [lq1_ms, lq2_ms, lq3_ms, gq3_ms] # adjusted to msk-1 so that the merge encountered for msk are of objects only present until dk-1
        mergeSecondarySizeFrac[msk - 1] = [x / y if y != 0 else 0 for x, y in zip(mergeSecondarySize[msk-1], all_t_by_size[msk-1])] # Fraction of objects in msk-1 merged

    for mpk in merge_primary_ids.keys():
        mask_mpt = np.isin(matrix[:, :, :, mpk - 2], merge_primary_ids[mpk])
        matrix_mptm = matrix[:, :, :, mpk - 2][mask_mpt]
        id_mp, count_mp = np.unique(matrix_mptm, return_counts=True)

        lq1_mp = sum(1 for element in count_mp if element <= q1)
        lq2_mp = sum(1 for element in count_mp if element > q1 and element <= q2)
        lq3_mp = sum(1 for element in count_mp if element > q2 and element <= q3)
        gq3_mp = sum(1 for element in count_mp if element > q3)

        mergePrimarySize[mpk - 1] = [lq1_mp, lq2_mp, lq3_mp,
                                       gq3_mp]  # adjusted to msk-1 so that the merge encountered for msk are of objects only present until dk-1
        mergePrimarySizeFrac[mpk - 1] = [x / y if y != 0 else 0 for x, y in zip(mergePrimarySize[mpk - 1],
                                                                                  all_t_by_size[
                                                                                      mpk - 1])]  # Fraction of objects in msk-1 merged

    mergeSecondarySizeDF = pd.DataFrame(list(mergeSecondarySizeFrac.values()))
    mergeSecondarySizeDF.columns = ['Merge as Secondary size <= q1', 'q1 < size <= q2', 'q2 < size <= q3', 'size > q3']

    mergePrimarySizeDF = pd.DataFrame(list(mergePrimarySizeFrac.values()))
    mergePrimarySizeDF.columns = ['Merge as Primary size <= q1', 'q1 < size <= q2', 'q2 < size <= q3', 'size > q3']

    print('      merge')

############################################################################################################################################################

    print('Life Expectancy')
    split_csv_ = pd.read_csv(os.path.join(plotsavepath, 'split_list.csv'))
    merge_csv_ = pd.read_csv(os.path.join(plotsavepath, 'merge_list.csv'))

    lq1, bq1q2, bq2q3, gq3 = {}, {}, {}, {}

    for t in range(matrix.shape[3]):
        flattened_matrix = matrix[:, :, :, t].flatten()
        unique_values, value_counts = np.unique(flattened_matrix, return_counts=True)
        unique_values = unique_values[1:]
        value_counts = value_counts[1:]

        lq1t, bq1q2t, bq2q3t, gq3t = [], [], [], []
        for index_, i_vc in enumerate(unique_values):
            if value_counts[index_] <= q1:
                lq1t.append(unique_values[index_])
            elif value_counts[index_] <= q2:
                bq1q2t.append(unique_values[index_])
            elif value_counts[index_] <= q3:
                bq2q3t.append(unique_values[index_])
            else:
                gq3t.append(unique_values[index_])
        lq1[t+1] = lq1t
        bq1q2[t+1] = bq1q2t
        bq2q3[t+1] = bq2q3t
        gq3[t+1] = gq3t

    ##################################

    splitlist = pd.read_csv(os.path.join(plotsavepath, 'split_list.csv'))
    splitted_ = splitlist['Splitted']
    splitted_into_ = splitlist['Splitted Into']

    all_id_and_progeny = {}

    for i_d in unique_values1:
        progeny_list = findProgeny(i_d, splitlist, [i_d], [])
        all_id_and_progeny[i_d] = progeny_list

    uniquexxx = {}
    for t in range(matrix.shape[3]):
        uniquexxx[t] = np.unique(matrix[:, :, :, t].flatten())

    last_t_of_all_ids_and_their_progeny = {}
    last_t_accounted_list = []

    for t in range(matrix.shape[3] - 1, -1, -1):
        current_list = []
        for i_d in all_id_and_progeny.keys():
            if set(all_id_and_progeny[i_d]).intersection(uniquexxx[t]) and i_d not in last_t_accounted_list:
                current_list.append(i_d)
                last_t_of_all_ids_and_their_progeny[i_d] = t
                last_t_accounted_list.append(i_d)

    last_t_of_all_ids_and_their_progeny_keys = list(last_t_of_all_ids_and_their_progeny.keys())
    last_t_of_all_ids_and_their_progeny_keys.sort()
    last_t_of_all_ids_and_their_progeny = {i: last_t_of_all_ids_and_their_progeny[i] for i in
                                           last_t_of_all_ids_and_their_progeny_keys}

    ##################################
    life_exp_lq1 = {}
    life_exp_bq1q2 = {}
    life_exp_bq2q3 = {}
    life_exp_gq3 = {}

    for t in range(matrix.shape[3] - 1):
        life_exp_lq1[t+1] = np.mean([last_t_of_all_ids_and_their_progeny[id_] - t + 1 for id_ in lq1[t+1]])
        life_exp_bq1q2[t+1] = np.mean([last_t_of_all_ids_and_their_progeny[id_] - t + 1 for id_ in bq1q2[t+1]])
        life_exp_bq2q3[t+1] = np.mean([last_t_of_all_ids_and_their_progeny[id_] - t + 1 for id_ in bq2q3[t+1]])
        life_exp_gq3[t+1] = np.mean([last_t_of_all_ids_and_their_progeny[id_] - t + 1 for id_ in gq3[t+1]])

        ##
    life_exp_lq1_new = {};
    life_exp_lq2_new = {};
    life_exp_lq3_new = {};
    life_exp_gq3_new = {}
    lq1_new = {};
    lq2_new = {};
    lq3_new = {};
    gq3_new = {}
    for k_ey in bornlist.keys():
            lq1_new[k_ey] = [value for value in lq1[k_ey] if value in bornlist[k_ey]]
            lq2_new[k_ey] = [value for value in bq1q2[k_ey] if value in bornlist[k_ey]]
            lq3_new[k_ey] = [value for value in bq2q3[k_ey] if value in bornlist[k_ey]]
            gq3_new[k_ey] = [value for value in gq3[k_ey] if value in bornlist[k_ey]]

    allbornlist = [value for value_list in bornlist.values() for value in value_list]
    last_t_of_all_ids_and_their_progeny_new = {i: last_t_of_all_ids_and_their_progeny[i] for i in
                                               last_t_of_all_ids_and_their_progeny_keys if i in allbornlist}

    for t in range(1, matrix.shape[3] - 1):
        life_exp_lq1_new[t+1] = np.mean([last_t_of_all_ids_and_their_progeny_new[id_] - t + 1 for id_ in lq1_new[t+1]])
        life_exp_lq2_new[t+1] = np.mean([last_t_of_all_ids_and_their_progeny_new[id_] - t + 1 for id_ in lq2_new[t+1]])
        life_exp_lq3_new[t+1] = np.mean([last_t_of_all_ids_and_their_progeny_new[id_] - t + 1 for id_ in lq3_new[t+1]])
        life_exp_gq3_new[t+1] = np.mean([last_t_of_all_ids_and_their_progeny_new[id_] - t + 1 for id_ in gq3_new[t+1]])

    life_exp_new = np.array([list(s) for (s) in list(zip(life_exp_lq1_new.values(),
                                                         life_exp_lq2_new.values(),
                                                         life_exp_lq3_new.values(),
                                                         life_exp_gq3_new.values()
                                                         )
                                                     )
                             ]
                            )

    life_exp_new_df = pd.DataFrame(life_exp_new,
                                   columns=['Life Expectancy NEW size <= q1', 'q1 < size <= q2', 'q2 < size <= q3', 'size > q3'])

    ##

    life_expectancies = np.array([list(s) for (s) in list(zip(life_exp_lq1.values(),
                                                              life_exp_bq1q2.values(),
                                                              life_exp_bq2q3.values(),
                                                              life_exp_gq3.values()))])

    life_expectancies_df = pd.DataFrame(life_expectancies,
                                        columns=['Life Expectancy All size <= q1', 'q1 < size <= q2', 'q2 < size <= q3', 'size > q3'])
    life_expectancies_df.to_csv(os.path.join(os.path.dirname(trackedimagepath), 'Life_expectancy_by_size.csv'),
                                index=False)

    print('      Expectancy')

    print('csv and plots')
    combined_df = pd.concat([split_by_size_df, mergePrimarySizeDF, mergeSecondarySizeDF,
                             born_by_size_df, dead_by_size_df, all_t_by_size_df, maskedIntensityBySize_df,
                             masked_intensity_df_,
                             life_expectancies_df, life_exp_new_df], axis=1)

    combined_df.to_csv(os.path.join(plotsavepath, 'Size_Distribution_of_Events_and_Expectancy_over_time_.csv'))

    fig, axes = plt.subplots(4, 5, figsize=(50, 40))

    dfs = [split_by_size_df, mergePrimarySizeDF, mergeSecondarySizeDF, born_by_size_df, dead_by_size_df,
           all_t_by_size_df,  # total_intensity_df_,
           maskedIntensityBySize_df,
           masked_intensity_df_, life_expectancies_df, life_exp_new_df]

    titles = ['Fraction of Objects of Each Quartile Undergoing Split Event',
              'Fraction of Objects of Each Quartile Undergoing Merge Event as "Primary"',
              'Fraction of Objects of Each Quartile Undergoing Merge Event as "Secondary"',
              'Number of Objects Born into a Quartile',
              'Fraction of Objects Dead from a Quartile',
              'Total Objects in Each Quartile',  # 'Total Image Intensity',
              'Masked Intensity By Size',
              'Object Intensity (masked total intensity)', 'Avg. Life Expectancy By Size',
              'Avg. Life Expectancy of New Objects By Size'
              ]

    y_axs = ['Fraction of Objects of the Same Group',
             'Fraction of Objects of the Same Group',
             'Fraction of Objects of the Same Group',
             'Number of Objects of the Same Group',
             'Fraction of Objects of the Same Group',
             'Number of Objects',  # 'Intensity Value',
             'Intensity of Each Group',
             'Intensity Value', 'Life Expectancy at Timepoints', 'Life Expectancy of New Obj.'
             ]

    colors = ['blue', 'purple', 'magenta', 'red']

    for _ in range(4):
        for __ in range(int(len(dfs) // 2 + len(dfs) % 2)):

            if _ < 2:
                ax = axes[_, __]
                if _ < 1 and __ < 3:
                    ax.set_ylim(0, 0.5)
                df = dfs[_ * 5 + __]
                # df.rolling(window=9, min_periods=1).mean().plot(ax=ax)
                df.plot(ax=ax, color=colors)
                # ax.axvline(x=acp, color='green', linestyle='--', label='apical constriction point')
                ax.set_title(titles[_ * 5 + __], fontsize=15)
                ax.set_xlabel('timepoint', fontsize=15)
                ax.set_ylabel(y_axs[_ * 5 + __], fontsize=15)
                ax.legend(fontsize=15)
            else:
                ax = axes[_, __]
                if _ < 3 and __ < 3:
                    ax.set_ylim(0, 0.5)
                df = dfs[(_ - 2) * 5 + __]
                df.rolling(window=21, min_periods=1).mean().plot(ax=ax, color=colors)
                # ax.axvline(x=acp, color='green', linestyle='--', label='apical constriction point')
                ax.set_title(titles[(_ - 2) * 5 + __] + ' (Moving Average)', fontsize=15)
                ax.set_xlabel('timepoint', fontsize=15)
                ax.set_ylabel(y_axs[(_ - 2) * 5 + __], fontsize=15)
                ax.legend(fontsize=15)
    plt.tight_layout()

    plt.savefig(os.path.join(plotsavepath, 'Size_Distribution_of_Events_and_Expectancy_over_Time.png'), facecolor='white')
    plt.close()
    print('      csv and plots')
    print('-------------------------------')










