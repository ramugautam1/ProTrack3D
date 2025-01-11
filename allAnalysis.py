import numpy as np
import pandas as pd
import os
import tifffile
import nibabel as nib
import matplotlib.pyplot as plt
import json
import sqlite3
from matplotlib.colors import LinearSegmentedColormap



colors=['blue','purple','magenta','red']
color5 = ['green']+ colors
color6 = ['orange','green','purple','magenta','cyan','red']

category = 'size'


def merge_dataframes_and_save(df_list, output_file):
    dflist=[]
    for df in df_list:
        if 't' in df.columns:
            # df = df.dropna()
            # print(df.head())
            df['t'] = df['t'].astype('Int64')  # Use 'Int64' for nullable integers
            mint = df['t'].min()
            maxt = df['t'].max()
            complete_range = pd.DataFrame({'t': range(int(mint), int(maxt + 1))})
            df_filled = pd.merge(complete_range, df, on='t', how='left')
            df_filled = df_filled.fillna(0)
            dflist.append(df_filled)
        else:
            dflist.append(df)

    # Merge the DataFrames
    merged_df = pd.concat(dflist, axis=1)

    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(output_file, index=False)

    print(f"Merged DataFrame saved to {output_file}")
    return merged_df


def getEventIntensityPlots(plotsavepath, sT, trackedimagepath, origImgPath):
    dbpath = os.path.dirname(trackedimagepath)
    samplename = os.path.basename(origImgPath)[:-4]

    #######################################################################

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
            SELECT 
                t,
                -- Counts for sizeQ = 1 (Q1)
                SUM(CASE WHEN sizeQ = 1 AND next_sizeQ > sizeQ THEN 1 ELSE 0 END) AS sizeInc_Q1,
                SUM(CASE WHEN sizeQ = 1 AND next_sizeQ = sizeQ THEN 1 ELSE 0 END) AS sizeEq_Q1,
                SUM(CASE WHEN sizeQ = 1 AND next_sizeQ < sizeQ THEN 1 ELSE 0 END) AS sizeDec_Q1,
                COUNT(CASE WHEN sizeQ = 1 THEN id END) AS count_Q1,  -- Count for sizeQ = 1

                -- Counts for sizeQ = 2 (Q2)
                SUM(CASE WHEN sizeQ = 2 AND next_sizeQ > sizeQ THEN 1 ELSE 0 END) AS sizeInc_Q2,
                SUM(CASE WHEN sizeQ = 2 AND next_sizeQ = sizeQ THEN 1 ELSE 0 END) AS sizeEq_Q2,
                SUM(CASE WHEN sizeQ = 2 AND next_sizeQ < sizeQ THEN 1 ELSE 0 END) AS sizeDec_Q2,
                COUNT(CASE WHEN sizeQ = 2 THEN id END) AS count_Q2,  -- Count for sizeQ = 2

                -- Counts for sizeQ = 3 (Q3)
                SUM(CASE WHEN sizeQ = 3 AND next_sizeQ > sizeQ THEN 1 ELSE 0 END) AS sizeInc_Q3,
                SUM(CASE WHEN sizeQ = 3 AND next_sizeQ = sizeQ THEN 1 ELSE 0 END) AS sizeEq_Q3,
                SUM(CASE WHEN sizeQ = 3 AND next_sizeQ < sizeQ THEN 1 ELSE 0 END) AS sizeDec_Q3,
                COUNT(CASE WHEN sizeQ = 3 THEN id END) AS count_Q3,  -- Count for sizeQ = 3

                -- Counts for sizeQ = 4 (Q4)
                SUM(CASE WHEN sizeQ = 4 AND next_sizeQ > sizeQ THEN 1 ELSE 0 END) AS sizeInc_Q4,
                SUM(CASE WHEN sizeQ = 4 AND next_sizeQ = sizeQ THEN 1 ELSE 0 END) AS sizeEq_Q4,
                SUM(CASE WHEN sizeQ = 4 AND next_sizeQ < sizeQ THEN 1 ELSE 0 END) AS sizeDec_Q4,
                COUNT(CASE WHEN sizeQ = 4 THEN id END) AS count_Q4   -- Count for sizeQ = 4

            FROM 
                object_properties
            WHERE 
                exists_next  
            GROUP BY 
                t
            ORDER BY 
                t;
        """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    conn.close()
    FlowObj = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    # splitObj.rolling(window=21, min_periods=1).mean().plot(color = color5, figsize=(10,6), title=f'Split')

    # Calculate fractions for sizeQ1
    FlowObj['Inc_Q1'] = np.where(FlowObj['count_Q1'] > 0, FlowObj['sizeInc_Q1'] / FlowObj['count_Q1'], 0)
    FlowObj['Eq_Q1'] = np.where(FlowObj['count_Q1'] > 0, FlowObj['sizeEq_Q1'] / FlowObj['count_Q1'], 0)
    FlowObj['Dec_Q1'] = np.where(FlowObj['count_Q1'] > 0, FlowObj['sizeDec_Q1'] / FlowObj['count_Q1'], 0)

    # Calculate fractions for sizeQ2
    FlowObj['Inc_Q2'] = np.where(FlowObj['count_Q2'] > 0, FlowObj['sizeInc_Q2'] / FlowObj['count_Q2'], 0)
    FlowObj['Eq_Q2'] = np.where(FlowObj['count_Q2'] > 0, FlowObj['sizeEq_Q2'] / FlowObj['count_Q2'], 0)
    FlowObj['Dec_Q2'] = np.where(FlowObj['count_Q2'] > 0, FlowObj['sizeDec_Q2'] / FlowObj['count_Q2'], 0)

    # Calculate fractions for sizeQ3
    FlowObj['Inc_Q3'] = np.where(FlowObj['count_Q3'] > 0, FlowObj['sizeInc_Q3'] / FlowObj['count_Q3'], 0)
    FlowObj['Eq_Q3'] = np.where(FlowObj['count_Q3'] > 0, FlowObj['sizeEq_Q3'] / FlowObj['count_Q3'], 0)
    FlowObj['Dec_Q3'] = np.where(FlowObj['count_Q3'] > 0, FlowObj['sizeDec_Q3'] / FlowObj['count_Q3'], 0)

    # Calculate fractions for sizeQ4
    FlowObj['Inc_Q4'] = np.where(FlowObj['count_Q4'] > 0, FlowObj['sizeInc_Q4'] / FlowObj['count_Q4'], 0)
    FlowObj['Eq_Q4'] = np.where(FlowObj['count_Q4'] > 0, FlowObj['sizeEq_Q4'] / FlowObj['count_Q4'], 0)
    FlowObj['Dec_Q4'] = np.where(FlowObj['count_Q4'] > 0, FlowObj['sizeDec_Q4'] / FlowObj['count_Q4'], 0)

    # Create separate DataFrames for each sizeQ category
    sizeQ1_df = FlowObj[['t', 'Inc_Q1', 'Eq_Q1', 'Dec_Q1']]
    sizeQ2_df = FlowObj[['t', 'Inc_Q2', 'Eq_Q2', 'Dec_Q2']]
    sizeQ3_df = FlowObj[['t', 'Inc_Q3', 'Eq_Q3', 'Dec_Q3']]
    sizeQ4_df = FlowObj[['t', 'Inc_Q4', 'Eq_Q4', 'Dec_Q4']]

    # Optionally, reset index if needed
    sizeQ1_df.reset_index(drop=True, inplace=True)
    sizeQ2_df.reset_index(drop=True, inplace=True)
    sizeQ3_df.reset_index(drop=True, inplace=True)
    sizeQ4_df.reset_index(drop=True, inplace=True)

    flowDFs = [sizeQ1_df, sizeQ2_df, sizeQ3_df, sizeQ4_df]

    titles = ['Q1', 'Q2', 'Q3', 'Q4']

    y_axs = ['Fraction of Objects',
             'Fraction of Objects',
             'Fraction of Objects',
             'Fraction of Objects',
             'Fraction of Objects',
             ]

    ix = 4
    jx = 1
    fig, axes = plt.subplots(ix, jx, figsize=(10 * jx, ix * 10))
    for i in range(ix):
        ax = axes[i]
        df = flowDFs[i].drop(columns=['t'])
        df.rolling(window=21, min_periods=1).mean().plot(kind='line', ax=ax,
                                                         color=['green', 'orange', 'red'])
        #                 ax.axvline(x=acp, color='green', linestyle='--', label='apical constriction point')
        ax.set_title(titles[i], fontsize=15)
        ax.set_xlabel('timepoint', fontsize=15)
        ax.set_ylabel(y_axs[i], fontsize=15)

    plt.tight_layout()
    plt.savefig(plotsavepath + '/ObjectFlowDiagram.png')
    plt.close()
    #######################################################################

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
             SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN 1 END, 0)) AS Q1Count,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN 1 END, 0)) AS Q2Count,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN 1 END, 0)) AS Q3Count,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN 1 END, 0)) AS Q4Count
                FROM
                    object_properties
                GROUP BY
                    t

    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # Close the connection
    conn.close()
    totObj = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    # totObj.drop(columns=['t']).rolling(window=21, min_periods=1).mean().plot(color = colors, figsize=(10,6), title=f'Total obj')
    totObj

    #######################################################################

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    event = 'split'

    query = f"""
        SELECT 
            t1.t,
            CASE WHEN t2.AllQ1Count <> 0 THEN t1.{event}Q1Count * 1.0 / t2.AllQ1Count ELSE 0 END AS Q1{event}Frac,
            CASE WHEN t2.AllQ2Count <> 0 THEN t1.{event}Q2Count * 1.0 / t2.AllQ2Count ELSE 0 END AS Q2{event}Frac,
            CASE WHEN t2.AllQ3Count <> 0 THEN t1.{event}Q3Count * 1.0 / t2.AllQ3Count ELSE 0 END AS Q3{event}Frac,
            CASE WHEN t2.AllQ4Count <> 0 THEN t1.{event}Q4Count * 1.0 / t2.AllQ4Count ELSE 0 END AS Q4{event}Frac
        FROM
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS {event}Q1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS {event}Q2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS {event}Q3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS {event}Q4Count
            FROM
                object_properties
            WHERE
                will_split_next and merge_primary_at_next_t=0
            GROUP BY
                t) AS t1
        INNER JOIN 
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS AllQ1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS AllQ2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS AllQ3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS AllQ4Count
            FROM
                object_properties
            GROUP BY
                t) AS t2
        ON t1.t = t2.t;
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # Close the connection
    conn.close()
    splitObj = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    # splitObj.rolling(window=21, min_periods=1).mean().plot(color = color5, figsize=(10,6), title=f'Split')
    splitObj

    #######################################################################

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    event = 'Merge'

    query = f"""
        SELECT 
            t1.t,
            CASE WHEN t2.AllQ1Count <> 0 THEN t1.{event}Q1Count * 1.0 / t2.AllQ1Count ELSE 0 END AS Q1{event}Frac,
            CASE WHEN t2.AllQ2Count <> 0 THEN t1.{event}Q2Count * 1.0 / t2.AllQ2Count ELSE 0 END AS Q2{event}Frac,
            CASE WHEN t2.AllQ3Count <> 0 THEN t1.{event}Q3Count * 1.0 / t2.AllQ3Count ELSE 0 END AS Q3{event}Frac,
            CASE WHEN t2.AllQ4Count <> 0 THEN t1.{event}Q4Count * 1.0 / t2.AllQ4Count ELSE 0 END AS Q4{event}Frac
        FROM
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS {event}Q1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS {event}Q2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS {event}Q3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS {event}Q4Count
            FROM
                object_properties
            WHERE
                (merge_primary_at_next_t or merge_secondary_at_next_t)
            GROUP BY
                t) AS t1
        INNER JOIN 
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS AllQ1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS AllQ2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS AllQ3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS AllQ4Count
            FROM
                object_properties
            GROUP BY
                t) AS t2
        ON t1.t = t2.t;
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # Close the connection
    conn.close()
    mergeObj = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    # mergeObj.rolling(window=21, min_periods=1).mean().plot(color = color5, figsize=(10,6), title=f'Merge')
    mergeObj

    #######################################################################
    #######################################################################

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    event = 'mergePrimary'

    query = f"""
        SELECT 
            t1.t,
            CASE WHEN t2.AllQ1Count <> 0 THEN t1.{event}Q1Count * 1.0 / t2.AllQ1Count ELSE 0 END AS Q1{event}Frac,
            CASE WHEN t2.AllQ2Count <> 0 THEN t1.{event}Q2Count * 1.0 / t2.AllQ2Count ELSE 0 END AS Q2{event}Frac,
            CASE WHEN t2.AllQ3Count <> 0 THEN t1.{event}Q3Count * 1.0 / t2.AllQ3Count ELSE 0 END AS Q3{event}Frac,
            CASE WHEN t2.AllQ4Count <> 0 THEN t1.{event}Q4Count * 1.0 / t2.AllQ4Count ELSE 0 END AS Q4{event}Frac
        FROM
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS {event}Q1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS {event}Q2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS {event}Q3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS {event}Q4Count
            FROM
                object_properties
            WHERE
                (merge_primary_at_next_t)
            GROUP BY
                t) AS t1
        INNER JOIN 
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS AllQ1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS AllQ2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS AllQ3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS AllQ4Count
            FROM
                object_properties
            GROUP BY
                t) AS t2
        ON t1.t = t2.t;
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # Close the connection
    conn.close()
    mergePrimObj = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    # mergeObj.rolling(window=21, min_periods=1).mean().plot(color = color5, figsize=(10,6), title=f'Merge')
    mergePrimObj

    #######################################################################

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    event = 'mergeSecondary'

    query = f"""
        SELECT 
            t1.t,
            CASE WHEN t2.AllQ1Count <> 0 THEN t1.{event}Q1Count * 1.0 / t2.AllQ1Count ELSE 0 END AS Q1{event}Frac,
            CASE WHEN t2.AllQ2Count <> 0 THEN t1.{event}Q2Count * 1.0 / t2.AllQ2Count ELSE 0 END AS Q2{event}Frac,
            CASE WHEN t2.AllQ3Count <> 0 THEN t1.{event}Q3Count * 1.0 / t2.AllQ3Count ELSE 0 END AS Q3{event}Frac,
            CASE WHEN t2.AllQ4Count <> 0 THEN t1.{event}Q4Count * 1.0 / t2.AllQ4Count ELSE 0 END AS Q4{event}Frac
        FROM
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS {event}Q1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS {event}Q2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS {event}Q3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS {event}Q4Count
            FROM
                object_properties
            WHERE
                (merge_secondary_at_next_t)
            GROUP BY
                t) AS t1
        INNER JOIN 
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS AllQ1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS AllQ2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS AllQ3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS AllQ4Count
            FROM
                object_properties
            GROUP BY
                t) AS t2
        ON t1.t = t2.t;
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # Close the connection
    conn.close()
    mergeSecObj = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    # mergeSecObj.rolling(window=21, min_periods=1).mean().plot(color = color5, figsize=(10,6), title=f'Merge')
    mergeSecObj

    #######################################################################
    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    event = 'death'
    query = f"""
        SELECT 
            t1.t,
            CASE WHEN t2.AllQ1Count <> 0 THEN t1.{event}Q1Count * 1.0 / t2.AllQ1Count ELSE 0 END AS Q1{event}Frac,
            CASE WHEN t2.AllQ2Count <> 0 THEN t1.{event}Q2Count * 1.0 / t2.AllQ2Count ELSE 0 END AS Q2{event}Frac,
            CASE WHEN t2.AllQ3Count <> 0 THEN t1.{event}Q3Count * 1.0 / t2.AllQ3Count ELSE 0 END AS Q3{event}Frac,
            CASE WHEN t2.AllQ4Count <> 0 THEN t1.{event}Q4Count * 1.0 / t2.AllQ4Count ELSE 0 END AS Q4{event}Frac
        FROM
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS {event}Q1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS {event}Q2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS {event}Q3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS {event}Q4Count
            FROM
                object_properties
            WHERE
                dead_after_t
            GROUP BY
                t) AS t1
        INNER JOIN 
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS AllQ1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS AllQ2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS AllQ3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS AllQ4Count
            FROM
                object_properties
            GROUP BY
                t) AS t2
        ON t1.t = t2.t;
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # Close the connection
    conn.close()
    deadObj = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    # deadObj.rolling(window=21, min_periods=1).mean().plot(color = color5, figsize=(10,6), title=f'Dead')
    deadObj

    #######################################################################
    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    event = 'birth'

    query = f"""
    SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS {event}Q1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS {event}Q2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS {event}Q3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS {event}Q4Count
            FROM
                object_properties
            WHERE
                born_at_t
            GROUP BY
                t
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # Close the connection
    conn.close()
    bornObj = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    # bornObj.rolling(window=21, min_periods=1).mean().plot(color = color5, figsize=(10,6), title=f'Dead')
    bornObj

    #######################################################################
    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
    SELECT
                t,
                SUM(CASE WHEN {category}Q = 1 THEN intensity END) AS Q1Intensity,
                SUM(CASE WHEN {category}Q = 2 THEN intensity END) AS Q2Intensity,
                SUM(CASE WHEN {category}Q = 3 THEN intensity END) AS Q3Intensity,
                SUM(CASE WHEN {category}Q = 4 THEN intensity END) AS Q4Intensity
            FROM
                object_properties
            GROUP BY
                t
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # Close the connection
    conn.close()
    intDF = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    # intDF.rolling(window=21, min_periods=1).mean().plot(color = color5, figsize=(10,6), title=f'Dead')
    intDF

    #######################################################################

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
    SELECT
                t,
                SUM(intensity)
            FROM
                object_properties
            GROUP BY
                t
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # Close the connection
    conn.close()
    totInt = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    # totInt.rolling(window=21, min_periods=1).mean().plot(color = color5, figsize=(10,6), title=f'Dead')
    totInt

    #######################################################################

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
    SELECT
                t,
                AVG(CASE WHEN {category}Q = 1 THEN remaining_life END) AS Q1Exp,
                AVG(CASE WHEN {category}Q = 2 THEN remaining_life END) AS Q2Exp,
                AVG(CASE WHEN {category}Q = 3 THEN remaining_life END) AS Q3Exp,
                AVG(CASE WHEN {category}Q = 4 THEN remaining_life END) AS Q4Exp
            FROM
                object_properties
            GROUP BY
                t
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # Close the connection
    conn.close()
    expDF = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    # expDF.rolling(window=21, min_periods=1).mean().plot(color = color5, figsize=(10,6), title=f'Dead')
    expDF

    #######################################################################

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
    SELECT
                t,
                AVG(CASE WHEN {category}Q = 1 THEN remaining_life END) AS Q1Exp,
                AVG(CASE WHEN {category}Q = 2 THEN remaining_life END) AS Q2Exp,
                AVG(CASE WHEN {category}Q = 3 THEN remaining_life END) AS Q3Exp,
                AVG(CASE WHEN {category}Q = 4 THEN remaining_life END) AS Q4Exp
            FROM
                object_properties
            WHERE
                born_at_t
            GROUP BY
                t
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # Close the connection
    conn.close()
    expDFnew = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    # expDFnew.rolling(window=21, min_periods=1).mean().plot(color = color5, figsize=(10,6), title=f'Dead')
    expDFnew
    #######################################################################

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    event = 'density'
    query = f"""
        SELECT 
                t,
                AVG(CASE WHEN {category}Q = 1 THEN density END) AS Q1{event},
                AVG(CASE WHEN {category}Q = 2 THEN density END) AS Q2{event},
                AVG(CASE WHEN {category}Q = 3 THEN density END) AS Q3{event},
                AVG(CASE WHEN {category}Q = 4 THEN density END) AS Q4{event}
            FROM
                object_properties
            GROUP BY
                t
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # Close the connection
    conn.close()
    densityDF = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    # densityDF.rolling(window=21, min_periods=1).mean().plot(color = color5, figsize=(10,6), title=f'Dead')
    densityDF

    #######################################

    #######################################################################

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    event = 'intensityInc'

    query = f"""
        SELECT 
            t1.t,
            CASE WHEN t2.AllQ1Count <> 0 THEN t1.{event}Q1Count * 1.0 / t2.AllQ1Count ELSE 0 END AS Q1{event}Frac,
            CASE WHEN t2.AllQ2Count <> 0 THEN t1.{event}Q2Count * 1.0 / t2.AllQ2Count ELSE 0 END AS Q2{event}Frac,
            CASE WHEN t2.AllQ3Count <> 0 THEN t1.{event}Q3Count * 1.0 / t2.AllQ3Count ELSE 0 END AS Q3{event}Frac,
            CASE WHEN t2.AllQ4Count <> 0 THEN t1.{event}Q4Count * 1.0 / t2.AllQ4Count ELSE 0 END AS Q4{event}Frac
        FROM
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS {event}Q1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS {event}Q2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS {event}Q3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS {event}Q4Count
            FROM
                object_properties
            WHERE
                intensity_change_next>0 and will_split_next=0 and merge_secondary_at_next_t=0 and merge_primary_at_next_t=0
            GROUP BY
                t) AS t1
        INNER JOIN 
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS AllQ1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS AllQ2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS AllQ3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS AllQ4Count
            FROM
                object_properties
            GROUP BY
                t) AS t2
        ON t1.t = t2.t;
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # Close the connection
    conn.close()
    intensityIncDF = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    # intensityIncDF.rolling(window=21, min_periods=1).mean().plot(color = color5, figsize=(10,6), title=f'Merge')
    intensityIncDF

    #######################################################################

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    event = 'intensityDec'

    query = f"""
        SELECT 
            t1.t,
            CASE WHEN t2.AllQ1Count <> 0 THEN t1.{event}Q1Count * 1.0 / t2.AllQ1Count ELSE 0 END AS Q1{event}Frac,
            CASE WHEN t2.AllQ2Count <> 0 THEN t1.{event}Q2Count * 1.0 / t2.AllQ2Count ELSE 0 END AS Q2{event}Frac,
            CASE WHEN t2.AllQ3Count <> 0 THEN t1.{event}Q3Count * 1.0 / t2.AllQ3Count ELSE 0 END AS Q3{event}Frac,
            CASE WHEN t2.AllQ4Count <> 0 THEN t1.{event}Q4Count * 1.0 / t2.AllQ4Count ELSE 0 END AS Q4{event}Frac
        FROM
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS {event}Q1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS {event}Q2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS {event}Q3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS {event}Q4Count
            FROM
                object_properties
            WHERE
                merge_primary_at_next_t=0 AND will_split_next=0 AND merge_secondary_at_next_t=0 AND dead_after_t=0 AND intensity_change_next<0

            GROUP BY
                t) AS t1
        INNER JOIN 
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS AllQ1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS AllQ2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS AllQ3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS AllQ4Count
            FROM
                object_properties
            GROUP BY
                t) AS t2
        ON t1.t = t2.t;
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # Close the connection
    conn.close()
    intensityDecDF = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    # intensityDecDF.rolling(window=21, min_periods=1).mean().plot(color = color5, figsize=(10,6), title=f'Merge')
    intensityDecDF

    #######################################################################

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    event = 'avgIntensity'

    query = f"""
        SELECT 
            t1.t,
            CASE WHEN t2.AllQ1Count <> 0 THEN t1.{event}Q1Count * 1.0 / t2.AllQ1Count ELSE 0 END AS Q1AvgInt,
            CASE WHEN t2.AllQ2Count <> 0 THEN t1.{event}Q2Count * 1.0 / t2.AllQ2Count ELSE 0 END AS Q2AvgInt,
            CASE WHEN t2.AllQ3Count <> 0 THEN t1.{event}Q3Count * 1.0 / t2.AllQ3Count ELSE 0 END AS Q3AvgInt,
            CASE WHEN t2.AllQ4Count <> 0 THEN t1.{event}Q4Count * 1.0 / t2.AllQ4Count ELSE 0 END AS Q4AvgInt
        FROM
            (SELECT
                t,
                SUM(CASE WHEN {category}Q = 1 THEN intensity END) AS {event}Q1Count,
                SUM(CASE WHEN {category}Q = 2 THEN intensity END) AS {event}Q2Count,
                SUM(CASE WHEN {category}Q = 3 THEN intensity END) AS {event}Q3Count,
                SUM(CASE WHEN {category}Q = 4 THEN intensity END) AS {event}Q4Count
            FROM
                object_properties
            WHERE
                intensity_change_next>0
            GROUP BY
                t) AS t1
        INNER JOIN 
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS AllQ1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS AllQ2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS AllQ3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS AllQ4Count
            FROM
                object_properties
            GROUP BY
                t) AS t2
        ON t1.t = t2.t;
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # Close the connection
    conn.close()
    avgIntensityDF = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    # avgIntensityDF.rolling(window=21, min_periods=1).mean().plot(color = color5, figsize=(10,6), title=f'Merge')
    avgIntensityDF

    #######################################################################

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    event = 'PeakInt'

    query = f"""
            SELECT
                t,
                AVG(CASE WHEN {category}Q = 1 THEN peak_intensity END) AS {event}Q1,
                AVG(CASE WHEN {category}Q = 2 THEN peak_intensity END) AS {event}Q2,
                AVG(CASE WHEN {category}Q = 3 THEN peak_intensity END) AS {event}Q3,
                AVG(CASE WHEN {category}Q = 4 THEN peak_intensity END) AS {event}Q4
            FROM
                object_properties
            GROUP BY
                t
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # Close the connection
    conn.close()
    peakIntDF = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    # peakIntDF.rolling(window=21, min_periods=1).mean().plot(color = color5, figsize=(10,6), title=f'Merge')
    peakIntDF

    #######################################################################
    #######################################################################

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    event = 'split'
    query = f"""
                select t1.t,  t1.splitcount* 1.0 / t2.totalcount as splitFrac
                from 
                 ( select t,count(*) as splitcount from object_properties where will_split_next and merge_primary_at_next_t=0 group by t ) as t1
                 inner join
                 (select t, count(*) as totalcount from object_properties group by t) as t2
                 on t1.t=t2.t
            """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # Close the connection
    conn.close()
    splitCombObj = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    # splitObj.rolling(window=21, min_periods=1).mean().plot(color = color5, figsize=(10,6), title=f'Split')
    splitCombObj

    #######################################################################

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    event = 'Merge'
    query = f"""
        SELECT 
            t1.t,
            t1.{event}Count * 1.0 / t2.AllCount AS {event}Frac
        FROM
            (SELECT
                t,
                COUNT(*) AS {event}Count
            FROM
                object_properties
            WHERE
             merge_primary_at_next_t or merge_secondary_at_next_t 
            GROUP BY
                t) AS t1
        INNER JOIN 
            (SELECT
                t,
                COUNT(*) AS AllCount
            FROM
                object_properties
            GROUP BY
                t) AS t2
        ON t1.t = t2.t;
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # Close the connection
    conn.close()
    mergeCombObj = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    # splitObj.rolling(window=21, min_periods=1).mean().plot(color = color5, figsize=(10,6), title=f'Split')
    mergeCombObj

    #######################################################################

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = """
                select t1.t, t1.total / t2.countt as average_intensity from

                (select t, sum(intensity) as total from object_properties group by t) as t1
                inner join
                
                (select t, count(*) as countt from object_properties group by t) as t2
                on t1.t=t2.t
            """
    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # Close the connection
    conn.close()
    avgIntObj = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    # splitObj.rolling(window=21, min_periods=1).mean().plot(color = color5, figsize=(10,6), title=f'Split')
    avgIntObj
    #######################################################################
    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
                        SELECT 
            t,
                AVG(CASE WHEN {category}Q = 1 THEN size END) AS {category}Q1_sizeAvg,
                AVG(CASE WHEN {category}Q = 2 THEN size END) AS {category}Q4_sizeAvg,
                AVG(CASE WHEN {category}Q = 3 THEN size END) AS {category}Q4_sizeAvg,
                AVG(CASE WHEN {category}Q = 4 THEN size END) AS {category}Q4_sizeAvg
            FROM 
                object_properties
            GROUP BY 
                t
            ORDER BY 
                t;
                """
    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # Close the connection
    conn.close()
    avgSizeObj = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    # splitObj.rolling(window=21, min_periods=1).mean().plot(color = color5, figsize=(10,6), title=f'Split')
    avgSizeObj
    #######################################################################
    dfs = [splitObj, mergeObj, mergePrimObj, mergeSecObj, deadObj, bornObj,
           totObj, totInt, intDF, peakIntDF, densityDF,
           avgIntensityDF, expDF, expDFnew, intensityIncDF, intensityDecDF,
           mergeCombObj, avgIntObj,avgSizeObj]
           #, splitCombObj,]
    output_file = plotsavepath + "/Size_Distribution_of_Events_and_Expectancy_over_Time.csv"
    merged_df = merge_dataframes_and_save(dfs, output_file)

    titles = ['Split Fraction', 'Total Merge Fraction', 'Primary merge Fraction', 'Secondary Merge Fraction',
              'Death Fraction', 'Birth Count',
              'Total Objects', 'Total Intensity', 'Group Intensity', 'Avg. Peak Intensity', 'Average Density',
              'Average Intensity', 'Avg. Life Expectancy', 'Avg. Life Expectancy of New Objects',
              'Fraction of objects with intensity increase', 'Fraction of objects with intensity decrease',
              'Merge Fraction, Overall', 'Average Intensity', 'Average Size'
              #, 'Split Fraction, Overall'
              ]

    y_axs = ['Fraction of Objects of the Same Group',
             'Fraction of Objects of the Same Group',
             'Fraction of Objects of the Same Group',
             'Fraction of Objects of the Same Group',
             'Fraction of Objects of the Same Group',
             'Number of Objects of the Same Group',

             'Number of Objects',
             'Total Intensity',
             'Total Intensity of Each Group',
             'Average Peak Intensity',
             'Average Density',

             'Average Intensity',
             'Life Expectancy in Timepoints',
             'Life Expectancy of New Obj.',
             'Fraction of Objects of the save group',
             'Fraction of Objects of the same group',
             'Fraction of All Objects',
             # 'Fraction of All Objects',
             'Average Intensity',
             'Average Size'
             ]

    colors = ['blue', 'purple', 'magenta', 'red']

    ix = 4;
    jx = 5
    fig, axes = plt.subplots(ix, jx, figsize=(10 * jx, ix * 10))
    for i in range(ix):
        for j in range(jx):
            if i > 2 and j > 3:
                pass
            else:
                ax = axes[i, j]
                if i < 1 and j < 4 and j > 0:
                    ax.set_ylim(0, 0.3)
                df = dfs[i * jx + j].drop(columns=['t'])
                df.rolling(window=21, min_periods=1).mean().plot(ax=ax, color=colors)
                #                 ax.axvline(x=acp, color='green', linestyle='--', label='apical constriction point')
                ax.set_title(titles[i * jx + j] + ' (Moving Average)', fontsize=15)
                ax.set_xlabel('timepoint', fontsize=15)
                ax.set_ylabel(y_axs[i * jx + j], fontsize=15)
                ax.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(plotsavepath + '/Size_Distribution_of_Events_and_Expectancy_over_Time.png')
    plt.close()
#######################################################################

    fractions1 = pd.concat([splitObj['Q1splitFrac'], mergeObj['Q1MergeFrac'], deadObj['Q1deathFrac'], intensityIncDF['Q1intensityIncFrac'], intensityDecDF['Q1intensityDecFrac']], axis=1)
    fractions1.columns = ['split', 'merge', 'death', 'stayInc', 'stayDec']

    fractions2 = pd.concat(
        [splitObj['Q2splitFrac'], mergeObj['Q2MergeFrac'], deadObj['Q2deathFrac'], intensityIncDF['Q2intensityIncFrac'], intensityDecDF['Q2intensityDecFrac']], axis=1)
    fractions2.columns = ['split', 'merge', 'death', 'stayInc', 'stayDec']

    fractions3 = pd.concat(
        [splitObj['Q3splitFrac'], mergeObj['Q3MergeFrac'], deadObj['Q3deathFrac'], intensityIncDF['Q3intensityIncFrac'], intensityDecDF['Q3intensityDecFrac']], axis=1)
    fractions3.columns = ['split', 'merge', 'death', 'stayInc', 'stayDec']

    fractions4 = pd.concat(
        [splitObj['Q4splitFrac'], mergeObj['Q4MergeFrac'], deadObj['Q4deathFrac'], intensityIncDF['Q4intensityIncFrac'], intensityDecDF['Q4intensityDecFrac']], axis=1)
    fractions4.columns = ['split', 'merge', 'death', 'stayInc', 'stayDec']


    dfs = [fractions1, fractions2, fractions3, fractions4]
    output_file = plotsavepath + "/Event_Frequency_Distribution.csv"
    merged_df = merge_dataframes_and_save(dfs, output_file)

    titles = ['Q1', 'Q2', 'Q3', 'Q4']

    y_axs = ['Fraction of Objects',
             'Fraction of Objects',
             'Fraction of Objects',
             'Fraction of Objects',
             'Fraction of Objects',
            ]

    ix = 4; jx=1
    fig, axes = plt.subplots(ix, jx, figsize=(10 * jx, ix * 10))
    for i in range(ix):
        ax = axes[i]
        df = dfs[i]
        df.rolling(window=21, min_periods=1).mean().plot(kind='line', ax=ax, color=['orange','green','red','blue','magenta'])
        #                 ax.axvline(x=acp, color='green', linestyle='--', label='apical constriction point')
        ax.set_title(titles[i] + ' (Moving Average)', fontsize=15)
        ax.set_xlabel('timepoint', fontsize=15)
        ax.set_ylabel(y_axs[i], fontsize=15)

    plt.tight_layout()
    plt.savefig(plotsavepath + '/Event_Frequency_Distribution.png')
#######################################################################

#     plt.show()

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################

def getIntensityChange(dbpath, sT, origImgPath, plotsavepath):
    trackedimagepath = os.path.join(dbpath, 'TrackedCombined.nii')
    samplename = os.path.basename(origImgPath)[:-4]

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
    SELECT 
                 t2.t as t,
                -t1.CondQ1Val + t2.CondQ1Val as CondQ1Val,
                -t1.CondQ2Val + t2.CondQ2Val as CondQ2Val,
                -t1.CondQ3Val + t2.CondQ3Val as CondQ3Val,
                -t1.CondQ4Val + t2.CondQ4Val as CondQ4Val
            FROM
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    merge_primary_at_next_t=0 AND will_split_next=0 AND merge_secondary_at_next_t=0 AND dead_after_t=0 AND intensity_change_next>0
                GROUP BY
                    t
                ) 
                AS 
                    t1
                INNER JOIN  
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    ((merge_primary_at_t=0) AND (splitted_at_t=0) AND (split_off_at_t=0) AND (born_at_t=0) AND intensity_change_from_prev>0)
                GROUP BY
                    t
                ) 
                AS 
                    t2
                ON 
                    t2.t = t1.t+1

    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    conn.close()

    IncBy = newDF[newDF['t'] > 2]

    ##################################################################
    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
    SELECT 
                 t2.t as t,
                -t1.CondQ1Val + t2.CondQ1Val as CondQ1Val,
                -t1.CondQ2Val + t2.CondQ2Val as CondQ2Val,
                -t1.CondQ3Val + t2.CondQ3Val as CondQ3Val,
                -t1.CondQ4Val + t2.CondQ4Val as CondQ4Val
            FROM
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    merge_primary_at_next_t=0 AND will_split_next=0 AND merge_secondary_at_next_t=0 AND dead_after_t=0 AND intensity_change_next<0
                GROUP BY
                    t
                ) 
                AS 
                    t1
                INNER JOIN  
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    (merge_primary_at_t=0) AND (splitted_at_t=0) AND (split_off_at_t=0) AND (born_at_t=0) AND intensity_change_from_prev<0
                GROUP BY
                    t
                ) 
                AS 
                    t2
                ON 
                    t2.t = t1.t+1

    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    conn.close()

    DecBy = newDF[newDF['t'] > 2]

    ##################################################################
    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
                SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    born_at_t
                GROUP BY
                    t
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    conn.close()

    BirthBy = newDF[newDF['t'] > 2]

    # ===========================================================================================================
    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
                SELECT
                    t+1 as t,
                    -SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    -SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    -SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    -SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    dead_after_t
                GROUP BY
                    t
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    conn.close()

    DeathBy = newDF[newDF['t'] > 2]
    DeathBy

    ##################################################################

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
                SELECT
                   t+1 as t,
                    SUM(intensity) AS Int
                FROM
                    object_properties
                WHERE
                    merge_secondary_at_next_t
                GROUP BY
                    t
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    conn.close()

    SecDF = newDF[newDF['t'] > 2]
    SecDF

    ##################################################################

    ##################################################################

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
                SELECT
                   t+1 as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity_change_next END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity_change_next END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity_change_next END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity_change_next END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    merge_primary_at_next_t and will_split_next=0
                GROUP BY
                    t
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    conn.close()

    MergeBy = newDF[newDF['t'] > 2]
    MergeBy

    ##################################################################
    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
                SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val,
                    SUM(intensity) AS TotalIntensity
                FROM
                    object_properties
                GROUP BY
                    t
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    conn.close()

    IntensityDF = newDF[newDF['t'] > 2].drop(columns=['t'])

    ##################################################################

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
            SELECT 
                t2.t as t,
                -t1.CondQ1Val + t2.CondQ1Val as CondQ1Val,
                -t1.CondQ2Val + t2.CondQ2Val as CondQ2Val,
                -t1.CondQ3Val + t2.CondQ3Val as CondQ3Val,
                -t1.CondQ4Val + t2.CondQ4Val as CondQ4Val
            FROM
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    will_split_next and merge_primary_at_next_t=0
                GROUP BY
                    t
                ) 
                AS 
                    t1
                INNER JOIN  
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    (splitted_at_t or split_off_at_t) and merge_primary_at_t=0
                GROUP BY
                    t
                ) 
                AS 
                    t2
                ON 
                    t1.t = t2.t-1

    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    conn.close()

    SplitBy = newDF[newDF['t'] > 2]

    #     print(SplitBy.loc[20:30,:])

    ##################################################################
    max_t = BirthBy['t'].max()
    # ==================GATHERING===============================================================
    dfs = [SplitBy, MergeBy, BirthBy, DeathBy, IncBy, DecBy]
    for df in dfs:
        df.fillna(0)
        df.set_index('t', inplace=True)
    for df in dfs:
        for i in range(3, max_t + 1):
            if i not in list(df.index):
                df.loc[i, :] = 0
    for df in dfs:
        df.sort_index(inplace=True)
    for df in dfs:
        df.astype(int)
    # ============
    #     SumBy = SplitBy + MergeBy + BirthBy + DeathBy + IncBy + DecBy
    # ===========================================================================================================

    SplitBy['CondQ4Val'].rolling(window=21, min_periods=1).mean(figsize=(10, 4)).plot()

    dfs = [SplitBy, MergeBy, BirthBy, DeathBy, IncBy, DecBy]
    alldf = pd.concat(
        [SplitBy.sum(axis=1), MergeBy.sum(axis=1), BirthBy.sum(axis=1), DeathBy.sum(axis=1), IncBy.sum(axis=1),
         DecBy.sum(axis=1)], axis=1)
    alldf = alldf.sum(axis=1).to_frame()

    output_file = plotsavepath + '/' + 'BY_' + category.upper() + '_change_in_Intensity.csv'
    merged_df = merge_dataframes_and_save(dfs, output_file)

    cumsum = alldf.cumsum()
    cumsum.columns = ['cumsum']

    alldf.columns = ['Intensity']
    alldf['Intensity'] = alldf['Intensity'] - SecDF['Int']
    alldf = alldf.rolling(window=21, min_periods=1).mean()

    titles = [
        'Intensity change of Split objects',
        'Intensity change of Merge objects',
        'Intensity of Born objects',
        'Intensity when dead',
        'Intensity change of Inactive, Increase',
        'Intensity change of Inactive, Decrease',

    ]

    y_axs = ['Intensity'] * len(titles)

    colors = ['blue', 'purple', 'magenta', 'red']

    ix = 2;
    jx = 3;
    countdf = 0
    fig, axes = plt.subplots(ix, jx, figsize=(15 * jx, ix * 10))
    for i in range(ix):
        for j in range(jx):
            if i * jx + j < len(dfs):
                countdf += 1
                ax = axes[i, j]
                df = dfs[i * jx + j]  # .drop(columns=['t'])
                df.rolling(window=21, min_periods=1).mean().plot(ax=ax, color=colors)
                ax.set_title(titles[i * jx + j] + ' (Moving Average)', fontsize=15)
                ax.set_xlabel('timepoint', fontsize=15)
                ax.set_ylabel(y_axs[i * jx + j], fontsize=15)
                ax.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(plotsavepath + '/' + 'BY_' + category.upper() + '_change_in_Intensity.png')
    #     plt.savefig(dbpath + '/Change_in_Intensity_due_to_various_events' +  '.png')
    #     plt.show()
    print('=================================================================================')

    #     print(SplitBy.loc[20:30,:])
    #     print(IncBy.loc[20:30,:])

    # Fraction is just a variable copied from somewhere, doesn't mean anything.
    Q = 'CondQ1Val'
    fractions1 = pd.concat([SplitBy[Q], MergeBy[Q], BirthBy[Q], DeathBy[Q], IncBy[Q], DecBy[Q]], axis=1)
    fractions1.columns = ['split', 'merge', 'birth', 'death', 'stayInc', 'stayDec']
    Q = 'CondQ2Val'
    fractions2 = pd.concat([SplitBy[Q], MergeBy[Q], BirthBy[Q], DeathBy[Q], IncBy[Q], DecBy[Q]], axis=1)
    fractions2.columns = ['split', 'merge', 'birth', 'death', 'stayInc', 'stayDec']
    Q = 'CondQ3Val'
    fractions3 = pd.concat([SplitBy[Q], MergeBy[Q], BirthBy[Q], DeathBy[Q], IncBy[Q], DecBy[Q]], axis=1)
    fractions3.columns = ['split', 'merge', 'birth', 'death', 'stayInc', 'stayDec']
    Q = 'CondQ4Val'
    fractions4 = pd.concat([SplitBy[Q], MergeBy[Q], BirthBy[Q], DeathBy[Q], IncBy[Q], DecBy[Q]], axis=1)
    fractions4.columns = ['split', 'merge', 'birth', 'death', 'stayInc', 'stayDec']

    f1 = fractions1.rolling(window=21, min_periods=1).mean()[11:-10]
    f2 = fractions2.rolling(window=21, min_periods=1).mean()[11:-10]
    f3 = fractions3.rolling(window=21, min_periods=1).mean()[11:-10]
    f4 = fractions4.rolling(window=21, min_periods=1).mean()[11:-10]

    dfs = [f1, f2, f3, f4, alldf, IntensityDF]

    ##########################################added####
    f1sum = f1.sum(axis=1)
    f2sum = f2.sum(axis=1)
    f3sum = f3.sum(axis=1)
    f4sum = f4.sum(axis=1)
    sumdfs = [f1sum, f2sum, f3sum, f4sum]
    ix = 4;
    jx = 1;
    countdf = 0
    sumtitle = ['Q1', 'Q2', 'Q3', 'Q4']
    fig, axes = plt.subplots(ix, jx, figsize=(20, 30))
    for i in range(ix):
        ax = axes[i]
        df = sumdfs[i]

        df.plot(ax=ax, kind='bar', color=colors[i])
        ax.axhline(y=0, c="black", linewidth=0.5, zorder=0)
        ax.set_ylabel('Total Change')
        ax.set_title(sumtitle[i])
        ax.set_xlabel('Time (t)')

        ax.legend(title='Total Change', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)
    plt.tight_layout()
    plt.savefig(
        plotsavepath + '/' + 'BY_' + category.upper() + '_change_in_Intensity__SUM__BAR.png')
    ##########################################added####

    title = ['Q1', 'Q2', 'Q3', 'Q4', 'Overall Change', 'Total Intensity']
    kinds = ['bar', 'bar', 'bar', 'bar', 'bar', 'line']

    ix = 6;
    jx = 1;
    countdf = 0
    fig, axes = plt.subplots(ix, jx, figsize=(20, 30))
    for i in range(ix):
        ax = axes[i]
        df = dfs[i]

        if i == 5:
            df.rolling(window=21, min_periods=1).mean().plot(ax=ax, color=['blue', 'purple', 'magenta', 'red', 'black'])
        else:
            df.plot(ax=ax, kind=kinds[i], stacked=True if i < 4 else False, color=color6 if i < 4 else 'black')
            ax.set_ylabel('Change Value')
        ax.set_title(title[i])
        ax.set_xlabel('Time (t)')

        ax.legend(title='Change', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)
    plt.tight_layout()
    plt.savefig(plotsavepath + '/' + 'BY_' + category.upper() + '_change_in_Intensity_BAR.png')

    ix = 4;
    jx = 1;
    countdf = 0
    fig, axes = plt.subplots(ix, jx, figsize=(20, 30))
    for i in range(ix):
        ax = axes[i]
        df = dfs[i]

        df.plot(ax=ax, kind='line', color=color6 if i < 4 else 'black')
        ax.axhline(y=0, c="black", linewidth=0.5, zorder=0)
        ax.set_ylabel('Change Value')
        ax.set_title(title[i])
        ax.set_xlabel('Time (t)')

        ax.legend(title='Change', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)
    plt.tight_layout()
    plt.savefig(plotsavepath + '/' + 'BY_' + category.upper() + '_change_in_Intensity_LINE.png')


####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################


def getIntensityChangeAvg(dbpath, sT, origImgPath, plotsavepath):
    trackedimagepath = os.path.join(dbpath, 'TrackedCombined.nii')
    samplename = os.path.basename(origImgPath)[:-4]

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
    SELECT 
                 t2.t as t,
                -t1.CondQ1Val + t2.CondQ1Val as CondQ1Val,
                -t1.CondQ2Val + t2.CondQ2Val as CondQ2Val,
                -t1.CondQ3Val + t2.CondQ3Val as CondQ3Val,
                -t1.CondQ4Val + t2.CondQ4Val as CondQ4Val
            FROM
                (SELECT
                    t as t,
                    AVG(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    AVG(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    AVG(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    AVG(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    merge_primary_at_next_t=0 AND will_split_next=0 AND merge_secondary_at_next_t=0 AND dead_after_t=0 AND intensity_change_next>0
                GROUP BY
                    t
                ) 
                AS 
                    t1
                INNER JOIN  
                (SELECT
                    t as t,
                    AVG(COALESCE(CASE WHEN previous_{category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    AVG(COALESCE(CASE WHEN previous_{category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    AVG(COALESCE(CASE WHEN previous_{category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    AVG(COALESCE(CASE WHEN previous_{category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    (merge_primary_at_t=0) AND (splitted_at_t=0) AND (split_off_at_t=0) AND (born_at_t=0) AND intensity_change_from_prev>0
                GROUP BY
                    t
                ) 
                AS 
                    t2
                ON 
                    t2.t = t1.t+1

    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    conn.close()

    IncBy = newDF[newDF['t'] > 2]

    ##################################################################
    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
    SELECT 
                 t2.t as t,
                -t1.CondQ1Val + t2.CondQ1Val as CondQ1Val,
                -t1.CondQ2Val + t2.CondQ2Val as CondQ2Val,
                -t1.CondQ3Val + t2.CondQ3Val as CondQ3Val,
                -t1.CondQ4Val + t2.CondQ4Val as CondQ4Val
            FROM
                (SELECT
                    t as t,
                    AVG(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    AVG(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    AVG(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    AVG(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    merge_primary_at_next_t=0 AND will_split_next=0 AND merge_secondary_at_next_t=0 AND dead_after_t=0 AND intensity_change_next<0
                GROUP BY
                    t
                ) 
                AS 
                    t1
                INNER JOIN  
                (SELECT
                    t as t,
                    AVG(COALESCE(CASE WHEN previous_{category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    AVG(COALESCE(CASE WHEN previous_{category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    AVG(COALESCE(CASE WHEN previous_{category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    AVG(COALESCE(CASE WHEN previous_{category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    (merge_primary_at_t=0) AND (splitted_at_t=0) AND (split_off_at_t=0) AND (born_at_t=0) AND intensity_change_from_prev<0
                GROUP BY
                    t
                ) 
                AS 
                    t2
                ON 
                    t2.t = t1.t+1

    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    conn.close()

    DecBy = newDF[newDF['t'] > 2]

    ##################################################################
    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
                SELECT
                    t as t,
                    AVG(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    AVG(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    AVG(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    AVG(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    born_at_t
                GROUP BY
                    t
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    conn.close()

    BirthBy = newDF[newDF['t'] > 2]

    # ===========================================================================================================
    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
                SELECT
                    t+1 as t,
                    -AVG(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    -AVG(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    -AVG(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    -AVG(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    dead_after_t
                GROUP BY
                    t
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    conn.close()

    DeathBy = newDF[newDF['t'] > 2]
    DeathBy

    ##################################################################

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
                SELECT
                   t+1 as t,
                    AVG(COALESCE(CASE WHEN {category}Q = 1 THEN intensity_change_next END, 0)) AS CondQ1Val,
                    AVG(COALESCE(CASE WHEN {category}Q = 2 THEN intensity_change_next END, 0)) AS CondQ2Val,
                    AVG(COALESCE(CASE WHEN {category}Q = 3 THEN intensity_change_next END, 0)) AS CondQ3Val,
                    AVG(COALESCE(CASE WHEN {category}Q = 4 THEN intensity_change_next END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    merge_primary_at_next_t and will_split_next=0
                GROUP BY
                    t
    """
    # and will_split_next=0

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    conn.close()

    MergeBy = newDF[newDF['t'] > 2]
    MergeBy

    ##################################################################
    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
                SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val,
                    SUM(intensity) AS TotalIntensity
                FROM
                    object_properties
                GROUP BY
                    t
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    conn.close()

    IntensityDF = newDF[newDF['t'] > 2].drop(columns=['t'])

    ##################################################################

    ##################################################################
    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()
    # using 2x the average for split, as the splitted and split_off are counted individually and will be divided
    # by twice the number of events -- yeeeeeaaaaah
    query = f"""
            SELECT
            tx.t as t,
            COALESCE(CASE WHEN t3.q1c>0 THEN tx.CondQ1Val/t3.q1c END, 0) as CondQ1Val,
            COALESCE(CASE WHEN t3.q2c>0 THEN tx.CondQ2Val/t3.q2c END, 0) as CondQ2Val,
            COALESCE(CASE WHEN t3.q3c>0 THEN tx.CondQ3Val/t3.q3c END, 0) as CondQ3Val,
            COALESCE(CASE WHEN t3.q4c>0 THEN tx.CondQ4Val/t3.q4c END, 0) as CondQ4Val

            FROM

            (SELECT 
                t2.t as t,
                (-t1.CondQ1Val + t2.CondQ1Val) as CondQ1Val,
                (-t1.CondQ2Val + t2.CondQ2Val) as CondQ2Val,
                (-t1.CondQ3Val + t2.CondQ3Val) as CondQ3Val,
                (-t1.CondQ4Val + t2.CondQ4Val) as CondQ4Val
            FROM
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    will_split_next and merge_primary_at_next_t=0
                GROUP BY
                    t
                ) 
                AS 
                    t1
                INNER JOIN  
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    (splitted_at_t or split_off_at_t) and merge_primary_at_t=0
                GROUP BY
                    t
                ) 
                AS 
                    t2
                ON 
                    t1.t = t2.t-1
                )
                AS 
                    tx
                INNER JOIN  
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN 1 END, 0)) AS q1c,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN 1 END, 0)) AS q2c,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN 1 END, 0)) AS q3c,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN 1 END, 0)) AS q4c
                FROM
                    object_properties
                WHERE
                    will_split_next and merge_primary_at_next_t=0
                GROUP BY
                    t
                ) 
                AS 
                    t3
                ON 
                    t3.t = tx.t-1

    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    conn.close()

    SplitBy = newDF[newDF['t'] > 2]

    SplitBy['CondQ4Val'].rolling(window=21, min_periods=1).mean().plot()
    #     stop

    #     print(SplitBy.loc[20:30,:])
    #     print(IncBy.loc[20:30,:])

    ##################################################################
    max_t = BirthBy['t'].max()
    # ==================GATHERING===============================================================
    dfs = [SplitBy, MergeBy, BirthBy, DeathBy, IncBy, DecBy]
    for df in dfs:
        df.fillna(0)
        df.set_index('t', inplace=True)
    for df in dfs:
        for i in range(3, max_t + 1):
            if i not in list(df.index):
                df.loc[i, :] = 0
    for df in dfs:
        df.sort_index(inplace=True)
    for df in dfs:
        df.astype(int)
    # ============
    #     SumBy = SplitBy + MergeBy + BirthBy + DeathBy + IncBy + DecBy
    # ===========================================================================================================

    dfs = [SplitBy, MergeBy, BirthBy, DeathBy, IncBy, DecBy]

    output_file = plotsavepath + '/' + 'BY_' + category.upper() + '_PER_EVENT_change_in_Intensity.csv'
    merged_df = merge_dataframes_and_save(dfs, output_file)

    titles = [
        'Avg. intensity change of Split objects',
        'Avg. intensity change of Merge objects',
        'Avg. intensity of Born objects',
        'Avg. intensity when dead',
        'Avg. intensity change of Inactive, Increase',
        'Avg. intensity change of Inactive, Decrease']

    y_axs = ['Intensity'] * len(titles)

    colors = ['blue', 'purple', 'magenta', 'red']

    ix = 2;
    jx = 3;
    countdf = 0
    fig, axes = plt.subplots(ix, jx, figsize=(15 * jx, ix * 10))
    for i in range(ix):
        for j in range(jx):
            if i * jx + j < len(dfs):
                countdf += 1
                ax = axes[i, j]
                df = dfs[i * jx + j]  # .drop(columns=['t'])
                df.rolling(window=21, min_periods=1).mean().plot(ax=ax, color=colors)
                #                 ax.axvline(x=acp, color='green', linestyle='--', label='apical constriction point')
                ax.set_title(titles[i * jx + j] + ' (Moving Average)', fontsize=15)
                ax.set_xlabel('timepoint', fontsize=15)
                ax.set_ylabel(y_axs[i * jx + j], fontsize=15)
                ax.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(plotsavepath + '/' + 'BY_' + category.upper() + '_PER_EVENT_change_in_Intensity.png')
    #     plt.savefig(dbpath + '/Change_in_Intensity_due_to_various_events' +  '.png')
    #     plt.show()

    ##################################################################

    # Fraction is just a variable copied from somewhere, doesn't mean anything.
    Q = 'CondQ1Val'
    fractions1 = pd.concat([SplitBy[Q], MergeBy[Q], BirthBy[Q], DeathBy[Q], IncBy[Q], DecBy[Q]], axis=1)
    fractions1.columns = ['split', 'merge', 'birth', 'death', 'stayInc', 'stayDec']
    Q = 'CondQ2Val'
    fractions2 = pd.concat([SplitBy[Q], MergeBy[Q], BirthBy[Q], DeathBy[Q], IncBy[Q], DecBy[Q]], axis=1)
    fractions2.columns = ['split', 'merge', 'birth', 'death', 'stayInc', 'stayDec']
    Q = 'CondQ3Val'
    fractions3 = pd.concat([SplitBy[Q], MergeBy[Q], BirthBy[Q], DeathBy[Q], IncBy[Q], DecBy[Q]], axis=1)
    fractions3.columns = ['split', 'merge', 'birth', 'death', 'stayInc', 'stayDec']
    Q = 'CondQ4Val'
    fractions4 = pd.concat([SplitBy[Q], MergeBy[Q], BirthBy[Q], DeathBy[Q], IncBy[Q], DecBy[Q]], axis=1)
    fractions4.columns = ['split', 'merge', 'birth', 'death', 'stayInc', 'stayDec']

    f1 = fractions1.rolling(window=21, min_periods=1).mean()[11:-10]
    f2 = fractions2.rolling(window=21, min_periods=1).mean()[11:-10]
    f3 = fractions3.rolling(window=21, min_periods=1).mean()[11:-10]
    f4 = fractions4.rolling(window=21, min_periods=1).mean()[11:-10]

    dfs = [f1, f2, f3, f4]
    title = ['Q1', 'Q2', 'Q3', 'Q4']

    ix = 4;
    jx = 1;
    countdf = 0
    fig, axes = plt.subplots(ix, jx, figsize=(20, 20))
    for i in range(ix):
        ax = axes[i]
        df = dfs[i]
        df.plot(ax=ax, kind='bar', stacked=True, color=color6)
        ax.set_title(title[i])
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Change Value')
        ax.legend(title='Change', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)
    plt.tight_layout()
    plt.savefig(
        plotsavepath + '/' + 'BY_' + category.upper() + '_PER_EVENT_change_in_Intensity_BAR.png')

    ix = 4;
    jx = 1;
    countdf = 0
    fig, axes = plt.subplots(ix, jx, figsize=(20, 30))
    for i in range(ix):
        ax = axes[i]
        df = dfs[i]

        df.plot(ax=ax, kind='line', color=color6 if i < 4 else 'black')
        ax.axhline(y=0, c="black", linewidth=0.5, zorder=0)
        ax.set_ylabel('Change Value')
        ax.set_title(title[i])
        ax.set_xlabel('Time (t)')

        ax.legend(title='Change', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)
    plt.tight_layout()
    plt.savefig(
        plotsavepath + '/' + 'BY_' + category.upper() + '_PER_EVENT_change_in_Intensity_LINE.png')


####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################

def getIntensityChangeContribution(dbpath, sT, origImgPath, plotsavepath):
    trackedimagepath = os.path.join(dbpath, 'TrackedCombined.nii')

    samplename = os.path.basename(origImgPath)[:-4]

    colors = ['blue', 'purple', 'magenta', 'red']
    color5 = ['green'] + colors

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
             SELECT
                    t as t,
                    SUM(CASE WHEN {category}Q = 1 THEN intensity END) AS CondQ1Val,
                    SUM(CASE WHEN {category}Q = 2 THEN intensity END) AS CondQ2Val,
                    SUM(CASE WHEN {category}Q = 3 THEN intensity END) AS CondQ3Val,
                    SUM(CASE WHEN {category}Q = 4 THEN intensity END) AS CondQ4Val
                FROM
                    object_properties
                GROUP BY
                    t

    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # Close the connection
    conn.close()
    TotalIntensity = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    #     TotalIntensity.rolling(window=21, min_periods=1).mean().plot(color = color5, figsize=(10,6), title=f'0911-04 Total Intensity')
    TotalIntensity

    # ===========================================================================================================

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
            SELECT t4.t as t,
                    t4.CondQ1Val - t5.CondQ1Val as CondQ1Val,
                    t4.CondQ2Val - t5.CondQ2Val as CondQ2Val,
                    t4.CondQ3Val - t5.CondQ3Val as CondQ3Val,
                    t4.CondQ4Val - t5.CondQ4Val as CondQ4Val
            FROM
                (SELECT 
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                GROUP BY
                    t
                ) 
                AS 
                    t4
                INNER JOIN
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                GROUP BY
                    t
                )
                AS 
                    t5
                ON
                    t4.t=t5.t+1

    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # Close the connection
    conn.close()
    RealBy = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    max_t = RealBy['t'].max()
    #     RealBy.rolling(window=21, min_periods=1).mean().plot(color = color5, figsize=(10,6), title=f'real intensity change')
    # RealBy=RealBy.drop(columns=['t'])
    RealBy, max_t

    # ===========================================================================================================

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
            SELECT 
                t2.t as t,
                -t1.CondQ1Val + t2.CondQ1Val as CondQ1Val,
                -t1.CondQ2Val + t2.CondQ2Val as CondQ2Val,
                -t1.CondQ3Val + t2.CondQ3Val as CondQ3Val,
                -t1.CondQ4Val + t2.CondQ4Val as CondQ4Val
            FROM
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    will_split_next and merge_primary_at_next_t=0
                GROUP BY
                    t
                ) 
                AS 
                    t1
                INNER JOIN  
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    (splitted_at_t or split_off_at_t) and merge_primary_at_t=0
                GROUP BY
                    t
                ) 
                AS 
                    t2
                ON 
                    t1.t = t2.t-1

    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # newDF=newDF.drop(columns=['t'])

    # Close the connection
    conn.close()
    color5 = ['green', 'blue', 'purple', 'magenta', 'red']
    colors = ['blue', 'purple', 'magenta', 'red']
    #     newDF.rolling(window=21, min_periods=1).mean().plot(color=color5, figsize=(10,6), title=f'intensity change due to split')
    SplitBy = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    SplitBy

    # ===========================================================================================================

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
            SELECT 
                t2.t as t,
                -t1.CondQ1Val + t2.CondQ1Val as CondQ1Val,
                -t1.CondQ2Val + t2.CondQ2Val as CondQ2Val,
                -t1.CondQ3Val + t2.CondQ3Val as CondQ3Val,
                -t1.CondQ4Val + t2.CondQ4Val as CondQ4Val
            FROM
                (SELECT
                   t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    (merge_primary_at_next_t or merge_secondary_at_next_t) and will_split_next=0
                GROUP BY
                    t
                ) 
                AS 
                    t1
                LEFT JOIN  
                (SELECT
                   t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    (splitted_at_t=0 and split_off_at_t=0) and merge_primary_at_t
                GROUP BY
                    t
                ) 
                AS 
                    t2
                ON 
                    t1.t = t2.t-1
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # newDF=newDF.drop(columns=['t'])

    # Close the connection
    conn.close()
    color5 = ['green', 'blue', 'purple', 'magenta', 'red']
    colors = ['blue', 'purple', 'magenta', 'red']
    #     newDF.rolling(window=21, min_periods=1).mean().plot(color=color5, figsize=(10,6), title=f'intensity change due to merge')
    MergeBy = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    MergeBy

    # ===========================================================================================================

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
            SELECT 
                t2.t as t,
                -t1.CondQ1Val + t2.CondQ1Val as CondQ1Val,
                -t1.CondQ2Val + t2.CondQ2Val as CondQ2Val,
                -t1.CondQ3Val + t2.CondQ3Val as CondQ3Val,
                -t1.CondQ4Val + t2.CondQ4Val as CondQ4Val
            FROM
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    merge_primary_at_next_t and will_split_next
                GROUP BY
                    t
                ) 
                AS 
                    t1
                INNER JOIN  
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    (splitted_at_t or split_off_at_t) and merge_primary_at_t
                GROUP BY
                    t
                ) 
                AS 
                    t2
                ON 
                    t1.t = t2.t-1

    """

    query = f"""
            -- Create a temporary table with a sequence of integers
            WITH RECURSIVE sequence AS (
                SELECT MIN(t) AS t
                FROM (
                    SELECT MIN(t) AS t FROM object_properties
                    UNION ALL
                    SELECT MIN(t) - 1 AS t FROM object_properties
                ) AS initial_values
                UNION ALL
                SELECT t + 1
                FROM sequence
                WHERE t < (
                    SELECT MAX(t)
                    FROM (
                        SELECT MAX(t) AS t FROM object_properties
                        UNION ALL
                        SELECT MAX(t) + 1 AS t FROM object_properties
                    ) AS max_values
                )
            ),

            t1 AS (
                SELECT
                    t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    merge_primary_at_next_t AND will_split_next
                GROUP BY
                    t
            ),

            t2 AS (
                SELECT
                    t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    (splitted_at_t OR split_off_at_t) AND merge_primary_at_t
                GROUP BY
                    t
            )

            SELECT 
                sequence.t,
                COALESCE(t2.CondQ1Val, 0) - COALESCE(t1.CondQ1Val, 0) AS CondQ1Val,
                COALESCE(t2.CondQ2Val, 0) - COALESCE(t1.CondQ2Val, 0) AS CondQ2Val,
                COALESCE(t2.CondQ3Val, 0) - COALESCE(t1.CondQ3Val, 0) AS CondQ3Val,
                COALESCE(t2.CondQ4Val, 0) - COALESCE(t1.CondQ4Val, 0) AS CondQ4Val
            FROM
                sequence
            LEFT JOIN
                t1 ON sequence.t = t1.t
            LEFT JOIN
                t2 ON sequence.t = t2.t - 1
            ORDER BY
                sequence.t;

    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # newDF=newDF.drop(columns=['t'])
    print(newDF)

    # Close the connection
    conn.close()
    #     newDF.rolling(window=21, min_periods=1).mean().plot(color=color5, figsize=(10,6), title=f'intensity change due to Double action')
    #     plt.show()
    DoubleBy = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    DoubleBy

    # ===========================================================================================================

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
                SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    born_at_t
                GROUP BY
                    t
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # newDF=newDF.drop(columns=['t'])

    # Close the connection
    conn.close()
    #     newDF.rolling(window=21, min_periods=1).mean().plot(color=color5, figsize=(10,6), title=f'intensity change due to born objects')
    #     plt.show()

    BirthBy = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    BirthBy

    # ===========================================================================================================
    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
                SELECT
                    t+1 as t,
                    -SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    -SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    -SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    -SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    dead_after_t
                GROUP BY
                    t
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # newDF=newDF.drop(columns=['t'])

    # Close the connection
    conn.close()
    #     newDF.rolling(window=21, min_periods=1).mean().plot(color=color5, figsize=(10,6), title=f' intensity change due to dead objects')
    #     plt.show()

    DeathBy = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    DeathBy

    # ===========================================================================================================

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
                SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    (previous_{category}Q != {category}Q) AND (merge_primary_at_t=0) AND (splitted_at_t=0) AND (split_off_at_t=0) AND (born_at_t=0)
                GROUP BY
                    t

    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # newDF=newDF.drop(columns=['t'])

    # Close the connection
    conn.close()
    #     newDF.rolling(window=21, min_periods=1).mean().plot(color=color5, figsize=(10,6), title=f' intensity change due to in-migrating objects')
    #     plt.show()
    MigrateInBy = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    MigrateInBy

    # ===========================================================================================================

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()
    #     category = 'size'

    query = f"""
                SELECT
                    t+1 as t,
                    -SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    -SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    -SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    -SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    next_{category}Q != {category}Q AND merge_primary_at_next_t=0 AND will_split_next=0 AND merge_secondary_at_next_t=0 AND dead_after_t=0
                GROUP BY
                    t
    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # newDF=newDF.drop(columns=['t'])

    # Close the connection
    conn.close()
    #     newDF.rolling(window=21, min_periods=1).mean().plot(color=color5, figsize=(10,6), title=f' intensity change due to out-migrating objects')
    #     plt.show()
    MigrateOutBy = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    MigrateOutBy

    # ===========================================================================================================

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    cursor = conn.cursor()

    query = f"""
    SELECT 
                t2.t as t,
                -t1.CondQ1Val + t2.CondQ1Val as CondQ1Val,
                -t1.CondQ2Val + t2.CondQ2Val as CondQ2Val,
                -t1.CondQ3Val + t2.CondQ3Val as CondQ3Val,
                -t1.CondQ4Val + t2.CondQ4Val as CondQ4Val
            FROM
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    next_{category}Q = {category}Q AND merge_primary_at_next_t=0 AND will_split_next=0 AND merge_secondary_at_next_t=0 AND dead_after_t=0
                GROUP BY
                    t
                ) 
                AS 
                    t1
                INNER JOIN  
                (SELECT
                   t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    previous_{category}Q = {category}Q AND merge_primary_at_t=0 AND splitted_at_t=0 AND split_off_at_t=0 AND born_at_t=0
                GROUP BY
                    t
                ) 
                AS 
                    t2
                ON 
                    t1.t = t2.t-1

    """

    cursor.execute(query)
    columns = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    newDF = pd.DataFrame(results)
    newDF.columns = columns
    # newDF=newDF.drop(columns=['t'])

    # Close the connection
    conn.close()
    #     newDF.rolling(window=21, min_periods=1).mean().plot(color=color5, figsize=(10,6), title=f'overall intensity change due to STAY objects')
    #     plt.show()
    StayBy = newDF[newDF['t'] > 2]  # .drop(columns=['t'])
    StayBy

    # ==================GATHERING===============================================================
    dfs = [SplitBy, MergeBy, DoubleBy, StayBy, MigrateInBy, MigrateOutBy, BirthBy, DeathBy]
    for df in dfs:
        df.fillna(0)
        df.set_index('t', inplace=True)
    for df in dfs:
        for i in range(3, max_t + 1):
            if i not in list(df.index):
                df.loc[i, :] = 0
    for df in dfs:
        df.sort_index(inplace=True)
    for df in dfs:
        df.astype(int)
    # ============

    # rows, cols = SplitBy.shape
    # # Create a new DataFrame with the same shape, filled with zeros
    # DoubleBy = pd.DataFrame(np.zeros((rows, cols)), columns=SplitBy.columns)

    SumBy = SplitBy + MergeBy + DoubleBy + StayBy + MigrateInBy + MigrateOutBy + BirthBy + DeathBy
    #     SumBy.rolling(window=21,min_periods=1).mean().plot( figsize=(10,6),color=colors, title='SUM') # ylim=(-1e6,1e6),
    RealBy.set_index('t', inplace=True)
    #     RealBy.rolling(window=21,min_periods=1).mean().plot( figsize=(10,6), color=colors, title='REAL') # ylim=(-1e6,1e6),
    #     plt.show()
    # ===========================================================================================================

    dfs = [TotalIntensity, SplitBy, MergeBy, DoubleBy, BirthBy, DeathBy, MigrateInBy, MigrateOutBy, StayBy, SumBy,
           RealBy]

    output_file = plotsavepath + '/' + '(CARE WHERE OBJS END UP)BY_' + category.upper() + '_CONTRIBUTION_TO_GROUP_INTENSITY_CHANGE.csv'
    merged_df = merge_dataframes_and_save(dfs, output_file)

    titles = ['Total Intensity',
              'Intensity change due to Split',
              'Intensity change due to Merge',
              'Intensity change due to Split + Merge',
              'Intensity change due to Birth',
              'Intensity change due to Death',
              'Intensity change due to Migrate In',
              'Intensity change due to Migrate Out',
              'Intensity change due to Stay',
              'SUM of Changes',
              'Real Change']

    y_axs = ['Intensity'] * len(titles)

    colors = ['blue', 'purple', 'magenta', 'red']

    ix = 3;
    jx = 4;
    countdf = 0
    fig, axes = plt.subplots(ix, jx, figsize=(15 * jx, ix * 10))
    for i in range(ix):
        for j in range(jx):
            if i * jx + j < len(dfs):
                countdf += 1
                ax = axes[i, j]
                df = dfs[i * jx + j]  # .drop(columns=['t'])
                df.rolling(window=21, min_periods=1).mean().plot(ax=ax, color=colors)
                #                 ax.axvline(x=acp, color='green', linestyle='--', label='apical constriction point')
                ax.set_title(titles[i * jx + j] + ' (Moving Average)', fontsize=15)
                ax.set_xlabel('timepoint', fontsize=15)

                ax.set_ylabel(y_axs[i * jx + j], fontsize=15)
                ax.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(
        plotsavepath + '/' + '(CARE WHERE OBJS END UP)BY_' + category.upper() + '_CONTRIBUTION_TO_GROUP_INTENSITY_CHANGE.png')


#     pd.concat([RealBy, SumBy, ], axis=1).to_csv('D:\\TrackTool\\LATEST\\BASEDONNEWGT\\ANALYSIS\\SizeDistribution_of_EventsAndIntensity\\sumandreal.csv')
#     pd.concat([TotalIntensity, SplitBy, MergeBy, DoubleBy, BirthBy, DeathBy, MigrateInBy, MigrateOutBy, StayBy, SumBy, RealBy], axis=1).to_csv('D:\\TrackTool\\LATEST\\BASEDONNEWGT\\ANALYSIS\\SizeDistribution_of_EventsAndIntensity\\alldfs.csv')
# pd.concatto_csv()
#     plt.savefig(dbpath + '/Change_in_Intensity_due_to_various_events' +  '.png')


def niftireadu32(arg):
    return np.asarray(nib.load(arg).dataobj).astype(np.uint32).squeeze()


def getHistogram(trackedimagepath, plotsavepath):
    image = niftireadu32(trackedimagepath)
    bin_edges = np.arange(5, 150, 3)  # Adjust bin range based on your data

    sizelist = []

    for t_ in range(image.shape[-1]):
        u, c = np.unique(image[:, :, :, t_], return_counts=True)
        c = c[1:]  # Exclude the count of 0 (background)
        sizelist.append(list(c))

    # Your data with time points
    data = sizelist

    # Initialize the plot
    plt.figure(figsize=(42, 6))

    # Define the colormap from blue to red
    n_bins = 150 / 5
    cmap_name = 'rainbow'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Iterate over each time point
    for time_index, values in enumerate(data):
        # Compute histogram for the current time point
        counts, _ = np.histogram(values, bins=bin_edges)

        # Create a scatter plot with color coding
        plt.scatter([time_index] * len(bin_edges[:-1]), bin_edges[:-1] + 0.5,
                    c=counts, cmap=cmap, marker='s', s=100, edgecolor='k', alpha=0.7)

    # Add color bar to show the count scale
    plt.colorbar(label='Count')

    # Customize the plot
    plt.xticks(range(len(data)), [f'Timepoint {i + 1}' for i in range(len(data))])
    plt.xlabel('Time Point')
    plt.ylabel('Value Range')
    plt.title('Size count histogram (3 pixel buckets (5-8,9-11, etc))')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(plotsavepath + '/' + 'sizeHistogram.png')


def getHistogramUno(trackedimagepath, plotsavepath):
    image = niftireadu32(trackedimagepath)
    sizelist = []

    for t_ in range(image.shape[-1]):
        u, c = np.unique(image[:, :, :, t_], return_counts=True)
        c = c[1:]
        sizelist.append(list(c))
    # print(sizelist)

    # Your data with time points
    data = sizelist

    # Initialize the plot
    plt.figure(figsize=(24, 6))

    # Define a colormap
    cmap = plt.get_cmap('rainbow')
    n_bins = image.shape[-1]
    cmap_name = 'rainbow'
    colors = ["Violet", "Blue", "Cyan", "Green", "Yellow", "Orange", "Red"]

    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    cmap = plt.get_cmap('rainbow')

    print(
        '################################################\n################################################\n################################################\n################################################\n################################################\n################################################\n')
    # Iterate over each time point
    for time_index, values in enumerate(data):
        # Count occurrences of each value
        unique, counts = np.unique(values, return_counts=True)

        # Create a scatter plot with color coding
        scatter = plt.scatter([time_index] * len(values), values,
                              c=[counts[np.where(unique == v)[0][0]] for v in values],
                              cmap=cmap, marker='s', s=10, edgecolor='none', alpha=0.7,
                              label=f'Timepoint {time_index + 1}')

    # Add color bar to show the count scale
    plt.colorbar(scatter, label='Count')

    # Customize the plot
    # plt.xticks(range(len(data)), [f'Timepoint {i + 1}' for i in range(len(data))])
    plt.xticks(
        [i for i in range(len(data)) if (i + 1) % 5 == 0],
        [f'{i + 1}' for i in range(len(data)) if (i + 1) % 5 == 0]
    )
    plt.xlabel('Time Point')
    plt.ylabel('Value')
    plt.title('Color coded scatter plots of size counts')
    # plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(plotsavepath + '/' + 'sizeHistogramNotBucket.png')

def all_analysis(trackedimagepath, sT, origImgPath):
    dbpath = os.path.dirname(trackedimagepath)
    plotsavepath = dbpath + '/Analysis0'
    if not os.path.isdir(plotsavepath):
        os.makedirs(plotsavepath)
    getEventIntensityPlots(plotsavepath, sT, trackedimagepath,origImgPath)
    getIntensityChangeContribution(dbpath,sT, origImgPath, plotsavepath)
    getIntensityChange(dbpath, sT, origImgPath,plotsavepath)
    getIntensityChangeAvg(dbpath, sT, origImgPath,plotsavepath)
    # getHistogram(trackedimagepath,plotsavepath)
    getHistogramUno(trackedimagepath, plotsavepath)

# def all_analysis_app(trackedimagepath, segPath, origImgPath):
#
#     segParams = pd.read_csv(segPath + '/segmentation_parameters.csv')
#     sT = segParams['startTime'][0]
#     dbpath = os.path.dirname(trackedimagepath)
#     plotsavepath = dbpath + '/Analysis'
#     if not os.path.isdir(plotsavepath):
#         os.makedirs(plotsavepath)
#     getEventIntensityPlots(plotsavepath, sT, trackedimagepath,origImgPath)
#     getIntensityChangeContribution(dbpath,sT, origImgPath, plotsavepath)
#     getIntensityChange(dbpath, sT, origImgPath,plotsavepath)
#     getIntensityChangeAvg(dbpath, sT, origImgPath,plotsavepath)
#     getHistogram(trackedimagepath,plotsavepath)
#     getHistogramUno(trackedimagepath, plotsavepath)