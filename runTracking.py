import os.path

from prepare import prepare
from trackStep1 import trackStep1
from trackStep2 import trackStep2

#this is for testing
from trying_to_come_up_with_my_own_algorithm import myTrackStep2
from testing_new_track_step_1 import myTrackStep1

# imgname.get(), segloc.get(), trackloc.get(), strT.get(), enT.get(),trbT.get(),ost.get(),p1n.get(),p2n.get()


def runTracking(imageName, segmentationOPFolder, trackingOPFolder, startTime, endTime, trackbackTime, min_obj_size, protein1Name, protein2Name):
    prepare(imageName, 'proteinA_', 'proteinB_')



    # trackStep1(segmentationOPFolder+'/',trackingOPFolder+'/',startTime,endTime, imageName=imageName)
    # trackStep2(track_op_folder=trackingOPFolder+'/', trackbackT=trackbackTime, startpoint=startTime, endpoint=endTime-1,
    #            protein1Name='proteinA_'+protein1Name, protein2Name='proteinB_'+protein2Name, imageName=imageName)

    myTrackStep1(segmentationOPFolder+'/',trackingOPFolder+'/',startTime,endTime, imageName=imageName)
    myTrackStep2(track_op_folder=trackingOPFolder + '/', trackbackT=trackbackTime, startpoint=startTime,
               endpoint=endTime - 1,
               protein1Name='proteinA_' + protein1Name, protein2Name='proteinB_' + protein2Name, imageName=imageName)

