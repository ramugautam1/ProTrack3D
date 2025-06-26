import numpy as np
import pandas as pd
import os
import tifffile
import nibabel as nib
import matplotlib.pyplot as plt
import json
import sqlite3
from scipy.ndimage import label, center_of_mass

def niftireadUint16(arg):
    return np.asarray(nib.load(arg).dataobj).astype(np.uint32).squeeze()

def getSizeQuartiles(image,matrix):
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
    q1, q2, q3 = int(np.percentile(countsall, 25)), int(np.percentile(countsall, 50)), int(np.percentile(countsall, 75))
    return q1,q2,q3


def runSizeIntensityAnalysis(dbpath, sT, trackedimagepath, origImgPath):
    '''
    Features saved for each object at every timepoint
        t INTEGER,
        id INTEGER,
        x INTEGER,
        y INTEGER,
        z INTEGER,
        size INTEGER,
        intensity BIGINT,
        peak_intensity BIGINT,
        density FLOAT,
        sizeQ TINYINT,
        intensityQ TINYINT,
        remaining_life SMALLINT,
        previous_sizeQ SMALLINT,
        previous_intensityQ SMALLINT,
        next_sizeQ SMALLINT,
        next_intensityQ SMALLINT,
        size_change_next SMALLINT,
        size_change_from_prev SMALLINT,
        intensity_change_next BIGINT,
        intensity_change_from_prev BIGINT,
        born_at_t BOOL,
        dead_after_t BOOL,
        splitted_at_t BOOL,
        split_off_at_t BOOL,
        merge_primary_at_t BOOL,
        merge_primary_at_next_t BOOL,
        merge_secondary_at_next_t BOOL,
        will_split_next BOOL,
        existed_previous BOOL,
        exists_next BOOL
    '''


    def findProgeny(i_d, splitlist, progeny_list, accounted_list):
        if i_d in splitted_:
            progeny_list.extend(splitlist.loc[splitlist['Splitted'] == i_d, 'Splitted Into'].tolist())
            accounted_list.append(i_d)
            for i_d_ in progeny_list:
                if i_d_ not in accounted_list:
                    findProgeny(i_d_, splitlist, progeny_list, accounted_list)
        return progeny_list

    image = tifffile.imread(origImgPath)
    image = np.transpose(image, (3, 2, 1, 0))

    countsavename = os.path.join(os.path.dirname(trackedimagepath), 'pixelCountPerObjectEveryTimepoint.csv')
    matrix = niftireadUint16(trackedimagepath)
    #     matrix=matrix[:,:,:,:10]

    matrix_t_size = matrix.shape[-1]
    image = image[:, :, :, sT:sT + matrix_t_size]

    # matrix=matrix[:,:,:,:10]
    # image=image[:,:,:,:10]

    matrix_shape = matrix.shape
    mask___ = np.zeros_like(matrix)
    mask___[matrix > 0] = 1
    masked_image = image * mask___

    flattened_matrix1 = matrix.flatten()
    unique_values1 = np.unique(flattened_matrix1, return_counts=False)

    all_ids = {}
    bornlist = {}
    deadlist = {}
    splitting_ids = {}
    split_into_ids = {}
    merge_primary_ids = {}
    merge_secondary_ids = {}

    with open(os.path.join(dbpath, 'all_id.json'), 'r') as f:
        all_ids = json.load(f)
    all_ids = {int(key): value for key, value in all_ids.items()}

    with open(os.path.join(dbpath, 'born_id.json'), 'r') as f:
        bornlist = json.load(f)
    bornlist = {int(key): value for key, value in bornlist.items()}
    with open(os.path.join(dbpath, 'dead_id.json'), 'r') as f:
        deadlist = json.load(f)
    deadlist = {int(key): value for key, value in deadlist.items()}

    with open(os.path.join(dbpath, 'splitting_id.json'), 'r') as f:
        splitting_ids = json.load(f)
    splitting_ids = {int(key): value for key, value in splitting_ids.items()}

    with open(os.path.join(dbpath, 'split_into_id.json'), 'r') as f:
        split_into_ids = json.load(f)
    split_into_ids = {int(key): value for key, value in split_into_ids.items()}

    with open(os.path.join(dbpath, 'truemerge_as_primary.json'), 'r') as f:
        merge_primary_ids = json.load(f)
    merge_primary_ids = {int(key): value for key, value in merge_primary_ids.items()}

    with open(os.path.join(dbpath, 'truemerge_as_secondary.json'), 'r') as f:
        merge_secondary_ids = json.load(f)
    merge_secondary_ids = {int(key): value for key, value in merge_secondary_ids.items()}

    splitlist = pd.read_csv(os.path.join(os.path.dirname(trackedimagepath), 'split_list.csv'))

    merge_list = pd.read_csv(os.path.join(os.path.dirname(trackedimagepath), 'merge_list.csv'))

    ####################################################################################
    frag_dict = {}
    for _, row in splitlist.iterrows():
        # Extract 'Splitted Into' value
        splitted_into = (row['Splitted Into'], row['Time'])
        other_values = row['Splitted']
        if splitted_into not in frag_dict.keys():
            frag_dict[splitted_into] = other_values
    ####################################################################################
    intensityDict = {}
    densityDict = {}
    sizeDict = {}
    allIntsAllT = []
    allDensAllT = []
    allTIdCountDict = {}
    print('Running Analysis...')
    Data = {}
    for t in range(1, matrix.shape[-1] + 1):
        Data[t] = {}

        t_id_count_dict = {}
        unique, count = np.unique(matrix[:, :, :, t - 1], return_counts=True)
        unique = unique[1:];
        count = count[1:]
        t_id_count_dict = {unique[i]: count[i] for i in range(len(unique))}
        for id_ in all_ids[t]:
            centroids_xyz = center_of_mass(matrix[:,:,:,t-1]==id_)
            xc, yc, zc = [round(xyzc) for xyzc in centroids_xyz]
            xdim, ydim, zdim, tdim = matrix.shape


            Data[t][id_] = {}

            int_1 = int(image[:, :, :, t - 1][matrix[:, :, :, t - 1] == id_].sum())
            peak_1 = int(image[:, :, :, t - 1][matrix[:, :, :, t - 1] == id_].max())
            allIntsAllT.append(int_1)
            intensityDict[(t, id_)] = int_1
            sze = int(t_id_count_dict[id_])
            densityDict[(t, id_)] = int_1 / sze
            sizeDict[(t, id_)] = sze

            Data[t][id_]["x"]=xc
            Data[t][id_]["y"]=yc
            Data[t][id_]["z"]=zc

            if xc<20 or xc>xdim-20:
                Data[t][id_]['edge_x'] = True
            else:
                Data[t][id_]['edge_x'] = False
            if yc<20 or yc>ydim-20:
                Data[t][id_]['edge_y'] = True
            else:
                Data[t][id_]['edge_y'] = False
            if zc < 1 or zc > zdim - 1:
                Data[t][id_]['edge_z'] = True
            else:
                Data[t][id_]['edge_z'] = False

            Data[t][id_]['peak_intensity'] = peak_1
            Data[t][id_]['size'] = sze
            Data[t][id_]['intensity'] = int(int_1)
            Data[t][id_]['density'] = int_1 / sze
        allTIdCountDict[t] = t_id_count_dict
        print(str(t), end=',')

    q1, q2, q3 = int(np.percentile(allIntsAllT, 25)), int(
        np.percentile(allIntsAllT, 50)), int(np.percentile(allIntsAllT, 75))

    sq1, sq2, sq3 = getSizeQuartiles(image, matrix)

    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print(q1, q2, q3, '\n', sq1, sq2, sq3)
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

    with open(os.path.join(dbpath, 'quartiles.txt'), 'w') as file:
        file.write(f"{sq1},{sq2},{sq3}")

    with open(os.path.join(dbpath, 'intensity_quartiles.txt'), 'w') as file:
        file.write(f"{q1},{q2},{q3}")

    all_id_and_progeny = {}
    splitted_ = splitlist['Splitted']
    splitted_into_ = splitlist['Splitted Into']

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
                last_t_of_all_ids_and_their_progeny[i_d] = t + 1
                last_t_accounted_list.append(i_d)
    last_t_of_all_ids_and_their_progeny_keys = list(last_t_of_all_ids_and_their_progeny.keys())
    last_t_of_all_ids_and_their_progeny_keys.sort()
    last_t_of_all_ids_and_their_progeny = {i: last_t_of_all_ids_and_their_progeny[i] for i in
                                           last_t_of_all_ids_and_their_progeny_keys}

    # for t in range(1,11):
    for t in all_ids.keys():
        if t <= matrix.shape[-1]:
            for id_ in all_ids[t]:
                idsize = sizeDict[(t, id_)]
                Data[t][id_]['sizeQ'] = 1 if idsize <= sq1 else 2 if idsize <= sq2 else 3 if idsize <= sq3 else 4

                idint = intensityDict[(t, id_)]
                Data[t][id_]['intensityQ'] = 1 if idint <= q1 else 2 if idint <= q2 else 3 if idint <= q3 else 4
                Data[t][id_]['progeny'] = all_id_and_progeny[id_]
                Data[t][id_]['remaining_life'] = int(last_t_of_all_ids_and_their_progeny[id_] - t + 1)

                Data[t][id_]['previous_sizeQ'] = Data[t - 1][id_]['sizeQ'] if (
                            t > 1 and (t - 1, id_) in intensityDict.keys()) else None
                Data[t][id_]['previous_intensityQ'] = Data[t - 1][id_]['intensityQ'] if (
                            t > 1 and (t - 1, id_) in intensityDict.keys()) else None

                if (id_, t) in frag_dict.keys():
                    tmpid = frag_dict[(id_, t)]
                    Data[t][id_]['previous_sizeQ'] = Data[t - 1][tmpid]['sizeQ']
                    Data[t][id_]['previous_intensityQ'] = Data[t - 1][tmpid]['intensityQ']

                Data[t][id_]['next_sizeQ'] = None
                Data[t][id_]['next_intensityQ'] = None
                if t > 1 and (t - 1, id_) in intensityDict.keys():
                    Data[t - 1][id_]['next_sizeQ'] = Data[t][id_]['sizeQ']
                    Data[t - 1][id_]['next_intensityQ'] = Data[t][id_]['intensityQ']

                sizeChangeNext = 0
                intensityChangeNext = 0
                sizeChangeFromPrev = 0
                intensityChangeFromPrev = 0

                intNow = intensityDict[(t, id_)]
                sizeNow = sizeDict[(t, id_)]
                if (t + 1, id_) in intensityDict.keys():
                    intNext = intensityDict[(t + 1, id_)]
                    intensityChangeNext = intNext - intNow
                    sizeChangeNext = sizeDict[(t + 1, id_)] - sizeDict[(t, id_)]
                else:
                    intensityChangeNext = -intNow;
                    sizeChangeNext = -sizeNow

                if (t - 1, id_) in intensityDict.keys():
                    intPrev = intensityDict[(t - 1, id_)]
                    intensityChangeFromPrev = intNow - intPrev
                    sizeChangeFromPrev = sizeNow - sizeDict[(t - 1, id_)]
                else:
                    intensityChangeFromPrev = intNow
                    sizeChangeFromPrev = sizeNow

                Data[t][id_]['size_change_next'] = sizeChangeNext
                Data[t][id_]['intensity_change_next'] = intensityChangeNext
                Data[t][id_]['size_change_from_prev'] = sizeChangeFromPrev
                Data[t][id_]['intensity_change_from_prev'] = intensityChangeFromPrev

                if t > 1:
                    if t < matrix.shape[-1]:
                        Data[t][id_]['born_at_t'] = True if id_ in bornlist[t] else False
                        Data[t][id_]['splitted_at_t'] = True if id_ in splitting_ids[t] else False
                        Data[t][id_]['split_off_at_t'] = True if id_ in split_into_ids[t] else False
                        Data[t][id_]['merge_primary_at_t'] = True if id_ in merge_primary_ids[t] else False

                        Data[t][id_]['merge_primary_at_next_t'] = True if id_ in merge_primary_ids[t + 1] else False
                        Data[t][id_]['merge_secondary_at_next_t'] = True if id_ in merge_secondary_ids[t + 1] else False
                        Data[t][id_]['dead_after_t'] = True if id_ in deadlist[t + 1] else False
                        Data[t][id_]['will_split_next'] = True if id_ in splitting_ids[t + 1] else False

                        Data[t][id_]['existed_previous'] = True if id_ in all_ids[t - 1] else False
                        Data[t][id_]['exists_next'] = True if id_ in all_ids[t + 1] else False

                    if t == matrix.shape[-1]:
                        Data[t][id_]['born_at_t'] = True if id_ in bornlist[t] else False
                        Data[t][id_]['splitted_at_t'] = True if id_ in splitting_ids[t] else False
                        Data[t][id_]['split_off_at_t'] = True if id_ in split_into_ids[t] else False
                        Data[t][id_]['merge_primary_at_t'] = True if id_ in merge_primary_ids[t] else False

                        Data[t][id_]['merge_primary_at_next_t'] = None
                        Data[t][id_]['merge_secondary_at_next_t'] = None
                        Data[t][id_]['dead_after_t'] = None
                        Data[t][id_]['will_split_next'] = None

                        Data[t][id_]['existed_previous'] = True if id_ in all_ids[t - 1] else False
                        Data[t][id_]['exists_next'] = None
                else:
                    Data[t][id_]['born_at_t'] = None
                    Data[t][id_]['dead_after_t'] = True if id_ in deadlist[t + 1] else False
                    Data[t][id_]['splitted_at_t'] = None
                    Data[t][id_]['split_off_at_t'] = None
                    Data[t][id_]['merge_primary_at_t'] = None
                    Data[t][id_]['merge_primary_at_next_t'] = True if id_ in merge_primary_ids[t + 1] else False
                    Data[t][id_]['merge_secondary_at_next_t'] = True if id_ in merge_secondary_ids[t + 1] else False
                    Data[t][id_]['will_split_next'] = True if id_ in splitting_ids[t + 1] else False

                    Data[t][id_]['existed_previous'] = None
                    Data[t][id_]['exists_next'] = True if id_ in all_ids[t + 1] else False

    import sqlite3

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS object_properties (
                t INTEGER, 
                id INTEGER,     
                x INTEGER,
                y INTEGER,
                z INTEGER,  
                size INTEGER, 
                edge_x BOOL,
                edge_y BOOL,
                edge_z BOOL,
                intensity BIGINT,
                peak_intensity BIGINT,
                density FLOAT,           
                sizeQ TINYINT,
                intensityQ TINYINT,          
                remaining_life SMALLINT,
                previous_sizeQ SMALLINT,
                previous_intensityQ SMALLINT,
                next_sizeQ SMALLINT,
                next_intensityQ SMALLINT,
                size_change_next SMALLINT,
                size_change_from_prev SMALLINT,
                intensity_change_next BIGINT,
                intensity_change_from_prev BIGINT,
                born_at_t BOOL,
                dead_after_t BOOL,
                splitted_at_t BOOL,
                split_off_at_t BOOL,
                merge_primary_at_t BOOL,
                merge_primary_at_next_t BOOL,
                merge_secondary_at_next_t BOOL,
                will_split_next BOOL,
                existed_previous BOOL,
                exists_next BOOL
            )

    ''')

    for t, data_items in Data.items():
        for id_, data in data_items.items():
            cursor.execute('''
                INSERT into object_properties (
                t,
                id,   
                x,
                y,
                z,    
                size, 
                edge_x,
                edge_y,
                edge_z,
                intensity, 
                peak_intensity,
                density,        
                sizeQ, 
                intensityQ,
                remaining_life,     
                previous_sizeQ,
                previous_intensityQ,
                next_sizeQ,
                next_intensityQ,  
                size_change_next, 
                size_change_from_prev,
                intensity_change_next,
                intensity_change_from_prev,
                born_at_t, 
                dead_after_t, 
                splitted_at_t, 
                split_off_at_t, 
                merge_primary_at_t, 
                merge_primary_at_next_t, 
                merge_secondary_at_next_t,
                will_split_next,
                existed_previous,
                exists_next

                )

                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)

            ''', (t,
                  id_,
                  int(data['x']),
                  int(data['y']),
                  int(data['z']),
                  int(data['size']),
                  data['edge_x'],
                  data['edge_y'],
                  data['edge_z'],
                  int(data['intensity']),
                  int(data['peak_intensity']),
                  float(data['density']),
                  int(data['sizeQ']),
                  int(data['intensityQ']),
                  int(data['remaining_life']),
                  data['previous_sizeQ'],
                  data['previous_intensityQ'],
                  data['next_sizeQ'],
                  data['next_intensityQ'],
                  int(data['size_change_next']),
                  int(data['size_change_from_prev']),
                  int(data['intensity_change_next']),
                  int(data['intensity_change_from_prev']),
                  data['born_at_t'],
                  data['dead_after_t'],
                  data['splitted_at_t'],
                  data['split_off_at_t'],
                  data['merge_primary_at_t'],
                  data['merge_primary_at_next_t'],
                  data['merge_secondary_at_next_t'],
                  data['will_split_next'],
                  data['existed_previous'],
                  data['exists_next']
                  )
                           )

    conn.commit()
    conn.close()
    print('\n=================Database Saved!===================')