import math

import numpy as np
import pandas as pd
from skimage import measure
import statistics

from functions import rand, size3, niftiwrite, niftiread, dashline, starline, niftiwriteF


def correlation(Fullsize_1, Fullsize_2, Fullsize_regression_1, Fullsize_regression_2,
                t2, time, spatial_extend_matrix, addr2, padding):
    # return
    Fullsize_1 = Fullsize_1.astype(int)
    Fullsize_2 = Fullsize_2.astype(int)

    depth = np.size(Fullsize_regression_1, axis=3)
    # get the size of sample
    [x, y, z] = size3(Fullsize_1)
    [x_reserve, y_reserve, z_reserve] = size3(Fullsize_1)
    # print([x, y, z])

    #padding the sample for 'extended search' (Fullsize: object label map, Fullsize_regression: object deep feature map)
    Fullsize_1_padding = np.pad(Fullsize_1, ((padding[0], padding[0]), (padding[1], padding[1]), (padding[2], padding[2])), 'constant')
    Fullsize_2_padding = np.pad(Fullsize_2, ((padding[0], padding[0]), (padding[1], padding[1]), (padding[2], padding[2])), 'constant')
    Fullsize_regression_1_padding = np.pad(Fullsize_regression_1, ((padding[0], padding[0]), (padding[1], padding[1]), (padding[2], padding[2]), (0, 0)), 'constant')
    Fullsize_regression_2_padding = np.pad(Fullsize_regression_2, ((padding[0], padding[0]), (padding[1], padding[1]), (padding[2], padding[2]), (0, 0)), 'constant')


    # correlation_map_padding=zeros(x+padding*2, y+padding*2, z+4,max(max(max(Fullsize_1))));
    correlation_map_padding_corr = np.zeros((x+padding[0]*2, y+padding[1]*2, z+padding[2]*2))
    correlation_map_padding_show = np.zeros((x+padding[0]*2, y+padding[1]*2, z+padding[2]*2))

    del Fullsize_regression_1, Fullsize_regression_2, Fullsize_1, Fullsize_2

    Fullsize_1_label = Fullsize_1_padding

    [fx, fy, fz] = size3(Fullsize_1_padding)

    stats1 = pd.DataFrame(measure.regionprops_table(Fullsize_1_padding, properties=('label', 'coords')))
    VoxelList = stats1.coords
    # print(VoxelList.shape[0])
    # print(stats1.shape[0])

    print('\nCalculating Correlation  ', end='')

    for i in range(0, stats1.shape[0]):

        VLi_size = np.size(VoxelList[i], axis=0)

        if i % (math.floor(stats1.shape[0]/50))==0:
            print('#', end='')

        if VLi_size < 30:
            stepsize = 1

        else:
            stepsize = 3

        # print(f'i {i}    stepsize    {stepsize}')

        # for a block of pixels in an object, search for the most correlated nearby block in previous time point
        for n1 in range(0, VLi_size, stepsize):
            if stepsize == 1:
                index = n1
            else:
                index = math.ceil(rand() * VLi_size)
            if index >= VLi_size:
                index = VLi_size - 1

            Feature_map1 = np.copy(Fullsize_regression_1_padding[
                           VoxelList[i][index][0]-3:VoxelList[i][index][0]+3+1,
                           VoxelList[i][index][1]-3:VoxelList[i][index][1]+3+1,
                           VoxelList[i][index][2]-1:VoxelList[i][index][2]+1+1,
                           :])

            for m1 in range(-1, 2):
                x = 2*m1
                for m2 in range(-1, 2):
                    y = 2*m2
                    for m3 in range(-1, 2):
                        z = m3
                        Feature_map2 = np.copy(Fullsize_regression_2_padding[
                                       VoxelList[i][index][0]+x-3:VoxelList[i][index][0]+x+3+1,
                                       VoxelList[i][index][1]+y-3:VoxelList[i][index][1]+y+3+1,
                                       VoxelList[i][index][2]+z-1:VoxelList[i][index][2]+z+1+1,
                                       :])

                        # *****************left to do
                        # ---uncomment if the extended search decay is wanted
                        # Feature_map2=Feature_map2.*spatial_extend_matrix;
                        # Feature_map1=Feature_map1/mean2(Feature_map1);
                        # Feature_map2=Feature_map2/mean2(Feature_map2);
                        # corr = convn(Feature_map1,Feature_map2(end:-1:1,end:-1:1,end:-1:1));

                        # # Flattening the feature map
                        Feature_map1_flatten = Feature_map1.flatten(order='F')
                        Feature_map2_flatten = Feature_map2.flatten(order='F')

                        # calculate correlation
                        corr = np.corrcoef(Feature_map1_flatten, Feature_map2_flatten)[0,1]
                        # print(f'{i}  {index} {corr}')

                        if corr > 0.2:
                            b = VoxelList[i]

                            a = []
                            for i1 in range(0, np.size(b,axis=0)):
                                # print(Fullsize_1_label[b[i1][0], b[i1][1], b[i1][2]])
                                a.append(Fullsize_1_label[b[i1][0], b[i1][1], b[i1][2]])

                            # print(f'a ------- {a}')

                            value = statistics.mode(np.array(a).flatten())

                            u, c = np.unique(np.array(a), return_counts=True)

                            try:
                                countZero = dict(zip(u, c))[0]
                            except:
                                countZero = 0


                            if countZero > value:
                                value = 0

                            correlation_map_padding_corr_local = correlation_map_padding_corr[
                                                                 VoxelList[i][index][0]+x-3:VoxelList[i][index][0]+x+3+1,
                                                                 VoxelList[i][index][1]+y-3:VoxelList[i][index][1]+y+3+1,
                                                                 VoxelList[i][index][2]+z-1:VoxelList[i][index][2]+z+1+1]

                            correlation_map_padding_show_local = correlation_map_padding_show[
                                                                 VoxelList[i][index][0]+x-3:VoxelList[i][index][0]+x+3+1,
                                                                 VoxelList[i][index][1]+y-3:VoxelList[i][index][1]+y+3+1,
                                                                 VoxelList[i][index][2]+z-1:VoxelList[i][index][2]+z+1+1]

                            # only select the highest correlation and assign the label

                            correlation_map_padding_show_local[correlation_map_padding_corr_local < corr] = value
                            correlation_map_padding_corr_local[correlation_map_padding_corr_local < corr] = corr

                            correlation_map_padding_corr[
                                                VoxelList[i][index][0]+x-3:VoxelList[i][index][0]+x+3+1,
                                                VoxelList[i][index][1]+y-3:VoxelList[i][index][1]+y+3+1,
                                                VoxelList[i][index][2]+z-1:VoxelList[i][index][2]+z+1+1] \
                                = correlation_map_padding_corr_local

                            correlation_map_padding_show[
                                                VoxelList[i][index][0]+x-3:VoxelList[i][index][0]+x+3+1,
                                                VoxelList[i][index][1]+y-3:VoxelList[i][index][1]+y+3+1,
                                                VoxelList[i][index][2]+z-1:VoxelList[i][index][2]+z+1+1] \
                                = correlation_map_padding_show_local

    niftiwrite(correlation_map_padding_show,
               addr2+'correlation_map_padding_show_traceback'+str(time)+'_'+t2 + '.nii')

    niftiwriteF(correlation_map_padding_corr,
                addr2 + 'correlation_map_padding_hide_traceback' + str(time) + '_' + t2 + '.nii')

    print('\nfiles saved.')


