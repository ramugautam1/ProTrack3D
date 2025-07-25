import os
from prepare import prepare
from myTrackStep2 import myTrackStep2
# from CTC_MyTrackStep2 import myTrackStep2
from myTrackStep1 import myTrackStep1
import glob
import time as theTime
from functions import niftireadu32

def runTracking(imageName, segmentationOPFolder, startTime, endTime, trackbackTime, min_obj_size, protein1Name, protein2Name):

    nowtime = theTime.perf_counter()

    trackingOPFolder__ = os.path.dirname(os.path.dirname(segmentationOPFolder))
    imgtagname = os.path.basename(imageName)[:-4]
    trackingOPFolder_ = trackingOPFolder__ + '/' + imgtagname + '_TrackingOutput'
    if not os.path.isdir(trackingOPFolder_):
        os.mkdir(trackingOPFolder_)
        print('created tracking output folder: \n', trackingOPFolder_)
    else:
        print('found tracking output folder: \n', trackingOPFolder_)

    imageName_ =  glob.glob(segmentationOPFolder+'/*.nii')[-1]

    prepare(imageName=imageName_, imageNameO=imageName, protein1Name='proteinA_', protein2Name='proteinB_')

    modelName = segmentationOPFolder.replace('\\', '/').split('/')[-1]
    myTrackStep1(segmentationOPFolder+'/',trackingOPFolder_+'/',startTime,endTime, imageNameS=imageName_, imageNameO=imageName)
    myTrackStep2(seg_op_folder = segmentationOPFolder+'/', track_op_folder=trackingOPFolder_ + '/', modelName=modelName, trackbackT=trackbackTime, startpoint=startTime,
               endpoint=endTime - 1,
               protein1Name='proteinA_' + protein1Name, protein2Name='proteinB_' + protein2Name, imageNameS=imageName_,imageNameO=imageName, stime=nowtime)

