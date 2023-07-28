import os
from prepare import prepare
from trackStep1 import trackStep1
from trackStep2 import trackStep2

#this is for testing
from myTrackStep2 import myTrackStep2
from myTrackStep1 import myTrackStep1
import glob
from functions import niftireadu32

# imgname.get(), segloc.get(), trackloc.get(), strT.get(), enT.get(),trbT.get(),ost.get(),p1n.get(),p2n.get()


def runTracking(imageName, segmentationOPFolder, startTime, endTime, trackbackTime, min_obj_size, protein1Name, protein2Name):


    trackingOPFolder__ = os.path.dirname(os.path.dirname(segmentationOPFolder))
    trackingOPFolder_ = trackingOPFolder__+'/TrackingOutput'
    if not os.path.isdir(trackingOPFolder_):
        os.mkdir(trackingOPFolder_)
        print('created tracking output folder: \n', trackingOPFolder_)
    else:
        print('found tracking output folder: \n', trackingOPFolder_)

    imageName_ =  glob.glob(segmentationOPFolder+'/*.nii')[-1]

    prepare(imageName=imageName_, imageNameO=imageName, protein1Name='proteinA_', protein2Name='proteinB_')

    # trackStep1(segmentationOPFolder+'/',trackingOPFolder_+'/',startTime,endTime, imageName=imageName)
    # trackStep2(track_op_folder=trackingOPFolder+'/', trackbackT=trackbackTime, startpoint=startTime, endpoint=endTime-1,
    #            protein1Name='proteinA_'+protein1Name, protein2Name='proteinB_'+protein2Name, imageName=imageName)
    # endTime = min(endTime,niftireadu32(imageName_).shape[-1])

    modelName = segmentationOPFolder.replace('\\', '/').split('/')[-1]
    myTrackStep1(segmentationOPFolder+'/',trackingOPFolder_+'/',startTime,endTime, imageNameS=imageName_, imageNameO=imageName)
    myTrackStep2(seg_op_folder = segmentationOPFolder+'/', track_op_folder=trackingOPFolder_ + '/', modelName=modelName, trackbackT=trackbackTime, startpoint=startTime,
               endpoint=endTime - 1,
               protein1Name='proteinA_' + protein1Name, protein2Name='proteinB_' + protein2Name, imageNameS=imageName_,imageNameO=imageName)

