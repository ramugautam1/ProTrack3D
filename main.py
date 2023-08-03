import tkinter as tk
# from PIL import Image, ImageTk
import tkinter.filedialog as fd
from runTracking import runTracking
import tkinter.font as font
from tkinter import ttk
from segment import segmentation, segmentation_predict
import os
# from familyTreesGenerator import generateFamilyTrees
from CompletelyNewFamilyTreeGen import generateFamilyTrees
from convert2nii import czi2nii, tif2nii

Font_tuple = ("Courier", 45, "bold")
Option_font_tuple = ("Courier", 20, "bold")


########################################################################################################################
########################################################################################################################
########################################################################################################################
class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        y = y + cy + self.widget.winfo_rooty() + 27
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("tahoma", "12", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


def CreateToolTip(widget, text):
    toolTip = ToolTip(widget)

    def enter(event):
        toolTip.showtip(text)

    def leave(event):
        toolTip.hidetip()

    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)


########################################################################################################################
########################################################################################################################
########################################################################################################################

window = tk.Tk()
window.rowconfigure(0, weight=1)
window.columnconfigure(0, weight=1)
window.wm_state('normal')

homepage = tk.Frame(window)
seg_page = tk.Frame(window)
track_page = tk.Frame(window)
fam_page = tk.Frame(window)
analysis_page = tk.Frame(window)
help_page = tk.Frame(window)
convert_page = tk.Frame(window)

train_page = tk.Frame(window)
test_page = tk.Frame(window)
predict_page = tk.Frame(window)
cross_corr_page = tk.Frame(window)

def_font = font.Font(family='System')
for frame in (homepage, seg_page, track_page, fam_page, help_page, train_page, test_page, predict_page, cross_corr_page,
              convert_page):
    frame.grid(row=0, column=0, sticky='nsew')


def show_frame(frame):
    frame.tkraise()


show_frame(homepage)

window.title('ProTrack3D - Segmentation, Tracking and Analysis')

window.geometry("1000x900")

my_menu = tk.Menu(window)
window.config(menu=my_menu)
my_menu.config(font=(def_font, 15))

file_menu = tk.Menu(my_menu)
ana_menu = tk.Menu(my_menu)

seg_menu = tk.Menu(my_menu)

# my_menu.add_cascade(label='File', menu=file_menu)
# my_menu.add_separator()
my_menu.add_command(label='Home', command=lambda: show_frame(homepage))
my_menu.add_separator()
my_menu.add_cascade(label='Segmentation', menu=seg_menu)  # , command=lambda: show_frame(seg_page))
#
seg_menu.add_command(label='Segmentation Home', font=('System', 15), command=lambda: show_frame(seg_page))
seg_menu.add_separator()
seg_menu.add_command(label='Train', font=('System', 15), command=lambda: show_frame(train_page))
# seg_menu.add_command(label='Test', command=lambda: show_frame(test_page), font=('System', 15))
seg_menu.add_command(label='Segment', font=('System', 15), command=lambda: show_frame(predict_page))
#
my_menu.add_separator()
my_menu.add_command(label='Tracking', command=lambda: show_frame(track_page))
my_menu.add_separator()
my_menu.add_command(label='Family Tree', command=lambda: show_frame(fam_page))
my_menu.add_separator()
# my_menu.add_command(label='Convert Files', command=lambda: show_frame(convert_page))
# my_menu.add_separator()
# my_menu.add_cascade(label='Analysis', menu=ana_menu)
# my_menu.add_separator()
my_menu.add_command(label='Help', command=lambda: show_frame(help_page))

file_menu.add_command(label='New', font=('System', 15))
file_menu.add_command(label='Open', font=('System', 15))
file_menu.add_command(label='Save', font=('System', 15))
file_menu.add_command(label='Save as', font=('System', 15))

# seg_menu.add_command(label='Segmentation',command=lambda: show_frame(seg_page))
# app_menu.add_command(label='Quit!', command=window.quit, background='blue')
# app_menu.add_command(label='Home', command=lambda: show_frame(homepage), background='green', font=('System', 15))

ana_menu.add_command(label='Cross-Correlation', font=('System', 15), command=lambda: show_frame(cross_corr_page))
ana_menu.add_command(label='Other Analysis', font=('System', 15))
ana_menu.add_command(label='Analysis III', font=('System', 15))

# ----------------------------------------------------------------------------------------------------------------------
# Homepage
homepage_greet = tk.Label(homepage, text='WELCOME TO ProTrack3D')
homepage_greet.configure(font=Font_tuple)
homepage_greet.place(x=50, y=100)
buttonH1 = tk.Button(homepage, text="Segmentation", width=15, command=lambda: show_frame(seg_page),
                     font=Option_font_tuple)

buttonH2 = tk.Button(homepage, text="Tracking", width=15,
                     command=lambda: show_frame(track_page), font=Option_font_tuple)
buttonH3 = tk.Button(homepage, width=15, text="Family Tree", command=lambda: show_frame(fam_page),
                     font=Option_font_tuple)
buttonH4 = tk.Button(homepage, width=15, text="Analysis", command=lambda: show_frame(analysis_page),
                     font=Option_font_tuple)
# buttonH5 = tk.Button(homepage, width=15, text = "Convert to .nii", \
# command=lambda : show_frame(convert_page), font=Option_font_tuple)
buttonH1.place(x=200, y=200)
buttonH2.place(x=200, y=300)
buttonH3.place(x=200, y=400)
buttonH4.place(x=200, y=500)
# buttonH5.place(x=200, y=600)

CreateToolTip(buttonH2, text='Click here to track objects in segmented images.!')
CreateToolTip(buttonH1,
              text='Click here for Segmentation!\nYou can proceed by training a model with your ground truth or '
                   '\nyou can use pre-trained models for segmentation.')
CreateToolTip(buttonH3, text='Click here to draw Family Trees using Tracking results!')
CreateToolTip(buttonH4, text='Click here for Analysis Menu!')
# CreateToolTip(buttonH5, text='Clich here to convert files to .nii format')

# ----------------------------------------------------------------------------------------------------------------------
# Segmentation page

seg_page_greet = tk.Label(seg_page, text='Segmentation', font=('Courier', 40, 'bold'))
seg_page_greet.place(x=50, y=120)

buttonS1 = tk.Button(seg_page, text="Back", command=lambda: show_frame(homepage), font=('System', 15))
buttonS2 = tk.Button(seg_page, width=15, text="Train", command=lambda: show_frame(train_page), font=('System', 16))
buttonS3 = tk.Button(seg_page, width=15, text="Segment", command=lambda: show_frame(predict_page), font=('System', 16))
buttonS4 = tk.Button(seg_page, width=15, text="Test", command=lambda: show_frame(track_page), font=('System', 16))

buttonS1.place(x=50, y=50)
buttonS2.place(x=50, y=200)
buttonS3.place(x=50, y=250)
# buttonS4.place(x=50, y=300)
CreateToolTip(buttonS2, text='Click to use your own data with ground truth to train a segmentation model.\n'
                             'If you want to use an already-trained model, please select Segment.')
CreateToolTip(buttonS3, text='Select to run segmentation on your data.')
############
predictOutputPath = tk.StringVar()
epochs = 0
imageName = tk.StringVar()
trainModelName = tk.StringVar()
predictModelName = tk.StringVar()
gt_path = tk.StringVar()
val_path = tk.StringVar()
train_output_location = tk.StringVar()


def browseGT():
    try:
        gt_path.set(fd.askdirectory())

    except:
        print('star!')
        gt_path.set('/default/gt/path/')


def browse_validation_data():
    try:
        val_path.set(fd.askdirectory())
    except:
        print('Not a Valid Path.')


def browseImage():
    imageName.set(fd.askopenfilename(defaultextension=".tif",
                                     filetypes=[("TIF Files", "*.tif"), ("NIFTI Files", "*.nii"),
                                                ("TIFF Files", "*.tiff")],
                                     initialdir='/home/nirvan/Desktop/AppTestRun'))
    # gt_path.set(imageName)


def trainOutputLocation():
    try:
        train_output_location.set(fd.askdirectory())

    except:
        train_output_location.set('/default/train/output/path/')


def predictOutputLocation():
    try:
        predictOutputPath.set(fd.askdirectory(initialdir='/home/nirvan/Desktop/AppTestRun'))
    except:
        predictOutputPath.set('/default/predict/output/path/')


def callTrain(modl, epochs, gt_p, op_p):
    print(modl, epochs, gt_p, op_p)
    segmentation.train(modl, epochs, gt_p, op_p)


def setPredictModelName(choice):
    predictModelName.set(choice)


def setTrainModelName(choice):
    trainModelName.set(choice)


##########################
train_page_greet = tk.Label(train_page, text='Train', font=('Courier', 40, 'bold'))
train_page_greet.place(x=50, y=120)

numEpochs = tk.IntVar()
label2 = tk.Label(train_page, text="No. of Epochs", font=('System', 15))
entry1 = tk.Entry(train_page, textvariable=numEpochs, font=('System', 15), width=14)
entry1.insert(0, '50')
button1 = tk.Button(train_page, text="Back", command=lambda: show_frame(seg_page), font=('System', 15))

button2 = tk.Button(train_page, text="Select Folder with Training Data", font=('System', 15),
                    command=lambda: browseGT())
entry2 = ttk.Entry(train_page, textvariable=gt_path, width=25, font=('System', 15))

button3 = tk.Button(train_page, text="Select Folder to Save Training Output", command=lambda: trainOutputLocation(),
                    font=('System', 15))
entry3 = ttk.Entry(train_page, textvariable=train_output_location, width=25, font=('System', 15))
#
modelNameList = ['FC-DenseNet', 'MobileUNet3D-Skip', 'ResNet-101', 'Encoder_Decoder3D', 'Encoder_Decoder3D_contrib',
                 'DeepLabV33D', 'FRRN-A', 'FRRN-B', 'FCN8', 'GCN-Res50', 'GCN-Res101', 'GCN-Res152', 'AdapNet3D',
                 'ICNet-Res50', 'ICNet-Res101', 'ICNet-Res152', 'PSPNet-Res50', 'PSPNet-Res101', 'PSPNet-Res152']

trainModelName.set(modelNameList[0])
modelMenu = tk.OptionMenu(train_page, trainModelName, *modelNameList, command=setTrainModelName)
modelMenu.config(font=('System', 15))

# buttonVal = tk.Button(train_page,text = "Select Folder with Validation Data", \
# font=('System',15), command=lambda : browse_validation_data())
# entryVal = tk.Entry(train_page, textvariable=val_path, width=25, font=('System',15))

#
button4 = tk.Button(train_page, width=10, text="Check",
                    command=lambda: print(
                        trainModelName.get() + '\n' + str(
                            numEpochs.get()) + '\n' + gt_path.get() + '\n' + train_output_location.get()),
                    font=('System', 15))
button5 = tk.Button(train_page, width=10, text="RUN", background="blue", foreground="white",
                    command=lambda: callTrain(trainModelName.get(), numEpochs.get(), gt_path.get(),
                                              train_output_location.get()),
                    font=('System', 15))
## dropdown
label1 = tk.Label(train_page, text='Model', font=('System', 15))
label1.place(x=50, y=200)

modelMenu.place(x=400, y=200)
##
button1.place(x=50, y=50)
label2.place(x=50, y=250)
entry1.place(x=500, y=250)
button2.place(x=50, y=300)
# button3.place(x=50, y=400)
# buttonVal.place(x=50, y=350)
# entryVal.place(x=500, y=350)

button4.place(x=50, y=500)
button5.place(x=50, y=550)

entry2.place(x=500, y=300)
# entry3.place(x=500, y=400)

CreateToolTip(train_page_greet, text='You can use your own data with ground truth to train a segmentation model.\n'
                                     'If you want to use an already-trained model, please go back and select Predict.')

CreateToolTip(label1, text='Select the Machine Learning model to use. Choices will be shown in the dropdown menu.')
CreateToolTip(label2, text='How many System do you want to train the model on training data? Default: 100')
CreateToolTip(button2, text='Select the folder that contains the training data.')
CreateToolTip(button3, text='Select the folder to save the outputs of training.')
CreateToolTip(button4, text='Click to check the console if everything you selected is correct.')
CreateToolTip(button5, text='Start training the selected model with your training data!')

# button6 = tk.Button(train_page, text='printModelName', command=lambda: print(modelName.get()))
# button6.place(x=400, y= 800)
###################################################
trainedModelPath = tk.StringVar()
trainedModel = tk.StringVar()
predict_output_path = tk.StringVar()


def getTrainedModel():
    trainedModelPath.set(fd.askdirectory(initialdir=os.getcwd() + '/checkpoints/' + predictModelName.get()))
    trainedModel.set(trainedModelPath.get() + "/latest_model_" + "_" + trainedModelPath.get().split('/')[-1] + '.ckpt')


def callPredict(model, image, startTime, endTime, trModelPath, predictOPpath):
    segmentation_predict.predict(model=model, image=image, startpoint=int(startTime), endpoint=int(endTime),
                                 modelCheckpointName=trModelPath, op_folder=predictOPpath)


predict_page_greet = tk.Label(predict_page, text='Segment', font=('Courier', 40, 'bold'))
predict_page_greet.place(x=50, y=120)

## dropdown
label1 = tk.Label(predict_page, text='Model', font=('System', 15))

modelNameList = ['FC-DenseNet', 'MobileUNet3D-Skip', 'ResNet-101', 'Encoder_Decoder3D', 'Encoder_Decoder3D_contrib',
                 'DeepLabV33D', 'FRRN-A', 'FRRN-B', 'FCN8', 'GCN-Res50', 'GCN-Res101', 'GCN-Res152', 'AdapNet3D',
                 'ICNet-Res50', 'ICNet-Res101', 'ICNet-Res152', 'PSPNet-Res50', 'PSPNet-Res101', 'PSPNet-Res152']

predictModelName.set('FC-DenseNet')
modelMenu = tk.OptionMenu(predict_page, predictModelName, *modelNameList, command=setPredictModelName)
modelMenu.config(font=('System', 15))

##

buttonPr1 = tk.Button(predict_page, text="Back", command=lambda: show_frame(seg_page), font=('System', 15))

buttonPr2 = tk.Button(predict_page, text="Select Image to Segment", command=lambda: browseImage(), font=('System', 15))
entryPr3 = tk.Entry(predict_page, textvariable=imageName, font=('System', 15))
buttonPr3 = tk.Button(predict_page, text="Select Folder to Save Segmentation Output",
                      command=lambda: predictOutputLocation(),
                      font=('System', 15))
entryPr4 = tk.Entry(predict_page, textvariable=predictOutputPath, font=('System', 15))
startT = tk.StringVar()
endT = tk.StringVar()
label2 = tk.Label(predict_page, text="Start Time", font=('System', 15))
entryPr1 = tk.Entry(predict_page, textvariable=startT, font=('System', 15), width=10)
entryPr1.insert(0, '1')
label3 = tk.Label(predict_page, text='End Time', font=('System', 15))
entryPr2 = tk.Entry(predict_page, textvariable=endT, font=('System', 15), width=10)
entryPr2.insert(0, '41')

buttonPr4 = tk.Button(predict_page, text="Select Folder with Trained Model", font=('System', 15),
                      command=lambda: getTrainedModel())
entryPr5 = tk.Entry(predict_page, textvariable=trainedModel, font=('System', 15))

buttonPr5 = tk.Button(predict_page, width=10, text="Check",
                      command=lambda: print(predictModelName.get(), imageName.get(), startT.get(),
                                            endT.get(), predictOutputPath.get()), font=('System', 15))
buttonPr6 = tk.Button(predict_page, width=10, text="RUN", background="blue", foreground="white",
                      command=lambda: callPredict(predictModelName.get(), imageName.get(), startT.get(), endT.get(),
                                                  trainedModel.get(), predictOutputPath.get()), font=('System', 15))

buttonPr1.place(x=50, y=50)

label1.place(x=50, y=200);
modelMenu.place(x=520, y=200)
label2.place(x=50, y=250);
entryPr1.place(x=520, y=250)
label3.place(x=50, y=300);
entryPr2.place(x=520, y=300)
buttonPr2.place(x=50, y=350);
entryPr3.place(x=520, y=350)
buttonPr4.place(x=50, y=450);
entryPr5.place(x=520, y=450)

buttonPr3.place(x=50, y=400);
entryPr4.place(x=520, y=400)

buttonPr5.place(x=50, y=600)
buttonPr6.place(x=50, y=650)

CreateToolTip(label2, text='SEnter the time frame to start segmentation. Default: 1')
CreateToolTip(label3, text='Enter the time frame to stop segmentation. Default: 41')
CreateToolTip(buttonPr2, text='Select the Image you want to run segmentation on.')
CreateToolTip(buttonPr3, text='Select the folder to save the outputs of segmentation.')
CreateToolTip(buttonPr5, text='Click to check the console if everything you selected is correct.')
CreateToolTip(buttonPr6, text='Start segmentation of your image.!')

################ TRACKING ###########################################################################################

imgname = tk.StringVar()
segloc = tk.StringVar()
trackloc = tk.StringVar()

# strT = tk.IntVar()
# enT = tk.IntVar()
# trbT = tk.IntVar()
# ost = tk.IntVar()
p1n = tk.StringVar()
p2n = tk.StringVar()
strT = tk.StringVar()
enT = tk.StringVar()
trbT = tk.StringVar()
ost = tk.StringVar()


def segOPfolder():
    segloc.set(fd.askdirectory())


def trackOPfolder():
    trackloc.set(fd.askdirectory())


def browseImageTr():
    imgname.set(fd.askopenfilename(defaultextension='.tif', filetypes=[("TIF Files", "*.tif"),
                                                                       ("TIFF Files", "*.tiff"),
                                                                       ("NIFTI Files", "*.nii")]))


# def callTracking():
#     tracking()


track_page_greet = tk.Label(track_page, text='Tracking', font=('Courier', 40, 'bold'))
track_page_greet.place(x=50, y=120)

buttonTr1 = tk.Button(track_page, text="Back", command=lambda: show_frame(homepage), font=('System', 15))

buttonTr2 = tk.Button(track_page, text="Select Original Images", command=lambda: browseImageTr(), font=('System', 15))
buttonTr3 = tk.Button(track_page, text="Select Folder with Segmentation Results", command=lambda: segOPfolder(),
                      font=('System', 15))
buttonTr4 = tk.Button(track_page, text="Select Folder to Save Tracking Results", command=lambda: trackOPfolder(),
                      font=('System', 15))
buttonTr5 = tk.Button(track_page, width=10, text="Check", font=('System', 15),
                      command=lambda: print("Check check check"))
buttonTr6 = tk.Button(track_page, width=10, text="RUN", font=('System', 15), background="blue", foreground="white",
                      command=lambda: runTracking(imageName=imgname.get(),
                                                  segmentationOPFolder=segloc.get(),
                                                  # trackingOPFolder=trackloc.get(),
                                                  startTime=int(strT.get()),
                                                  endTime=int(enT.get()), trackbackTime=int(trbT.get()),
                                                  min_obj_size=int(ost.get()),
                                                  protein1Name=p1n.get(),
                                                  protein2Name=p2n.get()))

labelTr1 = tk.Label(track_page, text="Start Time", font=('System', 15))
labelTr2 = tk.Label(track_page, text="Total Frames to Track", font=('System', 15))
labelTr3 = tk.Label(track_page, text="Min Size Threshold", font=('System', 15))
labelTr4 = tk.Label(track_page, text="Trackback Time", font=('System', 15))
# labelTr5 = tk.Label(track_page, text="Protein 1 Name", font=('System',15,'bold'))
# labelTr6 = tk.Label(track_page, text="Protein 2 Name", font=('System',15,'bold'))

entryTr1 = ttk.Entry(track_page, textvariable=imgname, font=('System', 15))
entryTr2 = ttk.Entry(track_page, textvariable=segloc, font=('System', 15))
entryTr3 = ttk.Entry(track_page, textvariable=trackloc, font=('System', 15))

entryTr4 = ttk.Entry(track_page, textvariable=strT, font=('System', 15))
entryTr4.insert(0, '1')
entryTr5 = ttk.Entry(track_page, textvariable=enT, font=('System', 15))
entryTr5.insert(0, '41')
entryTr6 = ttk.Entry(track_page, textvariable=trbT, font=('System', 15))
entryTr6.insert(0, '2')
entryTr7 = ttk.Entry(track_page, textvariable=ost, font=('System', 15))
entryTr7.insert(0, '6')
# entryTr8 = ttk.Entry(track_page, textvariable=p1n, font=('System',15,'bold'))
# entryTr9 = ttk.Entry(track_page, textvariable=p2n, font=('System',15,'bold'))

buttonTr1.place(x=50, y=50)
buttonTr2.place(x=50, y=200);
entryTr1.place(x=500, y=200, width=400)
buttonTr3.place(x=50, y=250);
entryTr2.place(x=500, y=250, width=400)
# buttonTr4.place(x=50, y=300);
# entryTr3.place(x=500, y=300, width=400)

# labelTr1.place(x=50, y=350);
# entryTr4.place(x=300, y=350, width=100)
labelTr2.place(x=50, y=400);
entryTr5.place(x=300, y=400, width=100)
labelTr3.place(x=50, y=450);
entryTr7.place(x=300, y=450, width=100)
labelTr4.place(x=50, y=500);
entryTr6.place(x=300, y=500, width=100)
# labelTr5.place(x=50, y=550); entryTr8.place(x=400, y=550, width=100)
# labelTr6.place(x=50, y=600); entryTr9.place(x=400, y=600, width=100)

buttonTr5.place(x=50, y=600, width=100)
buttonTr6.place(x=50, y=650, width=100)

CreateToolTip(buttonTr2, text='Select the image you want to track objects from.')
CreateToolTip(buttonTr3, text='Select the folder that contains the output of segmentation process.')
CreateToolTip(buttonTr4, text='Select a folder to save the output of tracking. \nThe folder must be empty.')
CreateToolTip(labelTr1, text='Select time frame to start traking (default 1).')
CreateToolTip(labelTr2, text='Select time frame to stop tracking (default 41).')
CreateToolTip(labelTr3, text='Select the minimum size of the objects to consider for tracking (default 27).')
CreateToolTip(labelTr4,
              text='Select trackback Time (How many time frames do you want to look for the objects to track (default 2).')

CreateToolTip(track_page_greet, text='This page is used for tracking the objects \n '
                                     'based on the original image and the segmentation output.\n'
                                     'You need to select the following:\n'
                                     '\t The image of interest\n'
                                     '\t The folder that contains the output of segmentation process\n'
                                     '\t An empty folder to save the output of tracking\n'
                                     '\t Time frame to start traking (default 1)\n'
                                     '\t Time frame to stop tracking (default 41)\n'
                                     '\t Minimum size of the objects to consider for tracking (default 27)\n'
                                     '\t Trackback Time (How many time frames to look for the objects to track (default 2).\n '
                                     'Tracking back more than 2 time frames SIGNIFICANTLY increases processing time'
              )

########################################################################################################################

test_page_greet = tk.Label(test_page, text='Test', font=('Courier', 40, 'bold'))
test_page_greet.place(x=50, y=120)
########################################################################################################################
# Family Tree

excelFile = tk.StringVar()
ftOpFolder = tk.StringVar()
branchMinLen = tk.StringVar()
objId = tk.StringVar()


def selectExcelFile():
    excelFile.set(fd.askopenfilename(defaultextension='.xlsx', filetypes=[("Excel Files", "*.xlsx")]))

def selectCsvFile():
    excelFile.set(fd.askopenfilename(defaultextension='.csv', filetypes=[("CSV Files", "*.csv")]))

def selectFtOutputFolder():
    ftOpFolder.set(fd.askdirectory())


# def generateFamilyTrees():
#     print("Processing Family Trees...")

def generateFamilyTree():
    print('Generating Family Tree for your object...')


fam_page_greet = tk.Label(fam_page, text='Family Tree and LPR', font=('Courier', 40, 'bold'))
fam_page_greet.place(x=50, y=120)

buttonFt1 = tk.Button(fam_page, text="Back", command=lambda: show_frame(homepage), font=('System', 15))

# buttonFt2 = tk.Button(fam_page, text="Select Tracking Results", command=lambda: selectExcelFile(), font=('System', 15))
buttonFt2 = tk.Button(fam_page, text="Select Tracking Results", command=lambda: selectCsvFile(), font=('System', 15))
entryFt2 = tk.Entry(fam_page, textvariable=excelFile, font=('System', 15))
buttonFt3 = tk.Button(fam_page, text="Select Folder to Save Family Trees", command=lambda: selectFtOutputFolder(),
                      font=('System', 15))
entryFt3 = tk.Entry(fam_page, textvariable=ftOpFolder, font=('System', 15))
buttonFt4 = tk.Button(fam_page, text='Branch Length Threshold', font=('System', 15))
entryFt4 = tk.Entry(fam_page, textvariable=branchMinLen, font=('System', 15))
entryFt4.insert(1, '5')

buttonFt6 = tk.Button(fam_page, text="Generate Family Tree", font=('System', 15), command=lambda: generateFamilyTree(),
                      background='blue', foreground='white')
buttonFt7 = tk.Button(fam_page, text="Generate All Family Trees",
                      command=lambda: generateFamilyTrees(excelFile=excelFile.get(), ftFolder=ftOpFolder.get()),
                      font=('System', 15), background="navy", foreground="white")
buttonFt8 = tk.Button(fam_page, text='Enter Object ID', font=('System', 15))
entryFt5 = tk.Entry(fam_page, textvariable=objId, font=('System', 15))

buttonFt1.place(x=50, y=50)
buttonFt2.place(x=50, y=200)
entryFt2.place(x=500, y=200, width=400)
buttonFt3.place(x=50, y=250)
entryFt3.place(x=500, y=250, width=400)
buttonFt4.place(x=50, y=300)
entryFt4.place(x=500, y=300, width=100)
buttonFt6.place(x=50, y=450)
buttonFt7.place(x=50, y=500)
buttonFt8.place(x=50, y=350)
entryFt5.place(x=500, y=350)

CreateToolTip(buttonFt2, text='Select the excel file that contains the tracking data.\n'
                              'You can find it in the folder you selected to save the tracking results.!')
CreateToolTip(buttonFt3,
              text='Click here to select a folder to store your family trees. \nPlease select an empty folder!')
CreateToolTip(buttonFt4,
              text='Select the minimum time frames an object has to exist \nto be considered for the family tree.')
CreateToolTip(buttonFt7, text='Click here to draw all the family trees!')
########################################################################################################################


# Analysis
# corss corr
ftOpFolderCC = tk.StringVar()
segOPfolderCC = tk.StringVar()
excelFileCC = tk.StringVar()


def selectSegmentationFolderCC():
    None


def selectExcelFileCC():
    None


def selectFTFolderCC():
    None


cross_corr_greet = tk.Label(cross_corr_page, text='Cross-Correlation Analysis', font=('Courier', 40, 'bold'))
cross_corr_greet.place(x=50, y=120)

buttonCC1 = tk.Button(cross_corr_page, text="Back", command=lambda: show_frame(homepage), font=('System', 15))

buttonCC2 = tk.Button(cross_corr_page, text="Select Segmentation Results Folder",
                      command=lambda: selectSegmentationFolderCC(), font=('System', 15))
entryCC2 = tk.Entry(cross_corr_page, textvariable=segOPfolderCC, font=('System', 15))
buttonCC3 = tk.Button(cross_corr_page, text="Select Tracking Results", command=lambda: selectExcelFileCC(),
                      font=('System', 15))

entryCC3 = tk.Entry(cross_corr_page, textvariable=excelFileCC, font=('System', 15))
buttonCC4 = tk.Button(cross_corr_page, text='Select Family Tree Folder', font=('System', 15),
                      command=lambda: selectFTFolderCC())
entryCC4 = tk.Entry(cross_corr_page, textvariable=ftOpFolderCC, font=('System', 15))
optionCC = tk.Label(cross_corr_page, borderwidth=1, relief='solid', text='Select Options to Run', font=('System', 15))
c1 = tk.Checkbutton(cross_corr_page, text='Cross Correlation Calculation', onvalue=1, font=('System', 15))
c2 = tk.Checkbutton(cross_corr_page, text='Statistical Analysis', onvalue=1, font=('System', 15))

buttonCC6 = tk.Button(cross_corr_page, width=10, text='RUN', font=('System', 15), background='blue', foreground='white')

buttonCC1.place(x=50, y=50)
buttonCC2.place(x=50, y=200);
entryCC2.place(x=500, y=200)
buttonCC3.place(x=50, y=250);
entryCC3.place(x=500, y=250)
buttonCC4.place(x=50, y=300);
entryCC4.place(x=500, y=300)
optionCC.place(x=50, y=350)
c1.place(x=70, y=400)
c2.place(x=70, y=450)
buttonCC6.place(x=50, y=550)
########################################################################################################################

help_page_greet = tk.Label(help_page, text='Help', font=('Courier', 40, 'bold'))
help_page_greet.place(x=50, y=120)

# ----------------------------------------------------------------------------------------------------------------------

filesFolder = tk.StringVar()
allFiles = tk.StringVar()


def selectFilesFolder():
    filesFolder.set(fd.askdirectory())


convert_page_greet = tk.Label(convert_page, text='Convert .czi and .tif to .nii', font=('Courier', 40, 'bold'))
buttonCon1 = tk.Button(convert_page, text="Back", command=lambda: show_frame(homepage), font=('System', 15))
buttonCon2 = tk.Button(convert_page, text="Select Folder with Files", command=lambda: selectFilesFolder(),
                       font=('System', 15))
entryCon1 = tk.Entry(convert_page, textvariable=filesFolder, font=('System', 15))

buttonCon3 = tk.Button(convert_page, text="Convert .czi to .nii", command=lambda: czi2nii(filesFolder.get()),
                       font=('System', 15))
buttonCon4 = tk.Button(convert_page, text="Convert .tif to .nii", command=lambda: tif2nii(filesFolder.get()),
                       font=('System', 15))

buttonCon1.place(x=50, y=50)
buttonCon2.place(x=50, y=100)
entryCon1.place(x=500, y=100)
buttonCon3.place(x=50, y=250)
buttonCon4.place(x=50, y=300)

# ----------------------------------------------------------------------------------------------------------------------

window.mainloop()
