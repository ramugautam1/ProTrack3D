import tkinter as tk
# from PIL import Image, ImageTk
import tkinter.filedialog as fd
from runTracking import runTracking
from allAnalysisApp import all_analysis_app
import tkinter.font as font
from tkinter import ttk
from segment import segmentation, segmentation_predict
import os
# # from familyTreesGenerator import generateFamilyTrees
# from CompletelyNewFamilyTreeGen import generateFamilyTrees
from FamilyTrees_WithSplitAndMerge import generateFamilyTrees
from FamilyTreeWithSplitAndMerge import generateFamilyTree
from convert2nii import czi2nii, tif2nii
import pandas as pd

Font_tuple = ("Courier", 45, "bold")
Option_font_tuple = ("Courier", 20, "bold")


# Font definitions
LARGE_FONT = ("Courier", 24, "bold")
MEDIUM_FONT = ("Courier", 16, "bold")
SMALL_FONT = ("Courier", 12)
BUTTON_FONT = ("System", 12)
ENTRY_FONT = ("System", 12)

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

    # def enter(event):
    #     toolTip.showtip(text)

    # def leave(event):
    #     toolTip.hidetip()

    # widget.bind('<Enter>', enter)
    # widget.bind('<Leave>', leave)

    widget.bind('<Enter>', lambda event: toolTip.showtip(text))
    widget.bind('<Leave>', lambda event: toolTip.hidetip())
    widget.bind('<Button-1>', lambda event: toolTip.hidetip())

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
              convert_page,analysis_page):
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
seg_menu.add_command(label='Segmentation Home', font=('System', 12), command=lambda: show_frame(seg_page))
seg_menu.add_separator()
seg_menu.add_command(label='Train', font=('System', 12), command=lambda: show_frame(train_page))
# seg_menu.add_command(label='Test', command=lambda: show_frame(test_page), font=('System', 12))
seg_menu.add_command(label='Segment', font=('System', 12), command=lambda: show_frame(predict_page))
#
my_menu.add_separator()
my_menu.add_command(label='Tracking', command=lambda: show_frame(track_page))
my_menu.add_separator()
my_menu.add_command(label='Family Tree', command=lambda: show_frame(fam_page))
my_menu.add_separator()
# my_menu.add_command(label='Convert Files', command=lambda: show_frame(convert_page))
# my_menu.add_separator()
my_menu.add_command(label='Analysis',  command=lambda: show_frame(analysis_page))
my_menu.add_separator()
my_menu.add_command(label='Help', command=lambda: show_frame(help_page))

file_menu.add_command(label='New', font=('System', 12))
file_menu.add_command(label='Open', font=('System', 12))
file_menu.add_command(label='Save', font=('System', 12))
file_menu.add_command(label='Save as', font=('System', 12))

# seg_menu.add_command(label='Segmentation',command=lambda: show_frame(seg_page))
# app_menu.add_command(label='Quit!', command=window.quit, background='blue')
# app_menu.add_command(label='Home', command=lambda: show_frame(homepage), background='green', font=('System', 12))

ana_menu.add_command(label='Cross-Correlation', font=('System', 12), command=lambda: show_frame(cross_corr_page))
ana_menu.add_command(label='Other Analysis', font=('System', 12))
ana_menu.add_command(label='Analysis III', font=('System', 12))

# ----------------------------------------------------------------------------------------------------------------------
# Homepage
# Get the screen width and height to calculate relative positions
screen_width = homepage.winfo_screenwidth()
screen_height = homepage.winfo_screenheight()

# Configure the homepage to expand with window resizing
homepage.pack_propagate(False)  # Prevent the frame from shrinking to fit contents
homepage.grid_propagate(False)

# Center the label
homepage_greet = tk.Label(homepage, text='ProTrack3D')
homepage_greet.configure(font=Font_tuple)
homepage_greet.place(relx=0.5, rely=0.1, anchor='center')

# Create buttons with consistent styling
buttons = [
    ("Segmentation", lambda: show_frame(seg_page),
     "Click here for Segmentation!\nYou can proceed by training a model with your ground truth or \nyou can use pre-trained models for segmentation."),
    ("Tracking", lambda: show_frame(track_page),
     "Click here to track objects in segmented images!"),
    ("Family Tree", lambda: show_frame(fam_page),
     "Click here to draw Family Trees using Tracking results!"),
    ("Analysis", lambda: show_frame(analysis_page),
     "Click here for Analysis Menu!")
]

# Place buttons vertically centered with equal spacing
for i, (text, command, tooltip) in enumerate(buttons):
    btn = tk.Button(homepage, text=text, width=20, command=command, font=Option_font_tuple)
    btn.place(relx=0.5, rely=0.3 + i*0.15, anchor='center')
    CreateToolTip(btn, text=tooltip)

# ----------------------------------------------------------------------------------------------------------------------
# Segmentation page

# Centered label
seg_page_greet = tk.Label(seg_page, text='Segmentation', font=('Courier', 40, 'bold'))
seg_page_greet.place(relx=0.5, rely=0.2, anchor='center')

# Back button (top-left corner)
buttonS1 = tk.Button(seg_page, text="Back", command=lambda: show_frame(homepage), font=('System', 12))
buttonS1.place(relx=0.05, rely=0.05, anchor='nw')  # 5% from left, 5% from top

# Main action buttons centered vertically
buttons = [
    ("Train", lambda: show_frame(train_page),
     'Click to use your own data with ground truth to train a segmentation model.\n'
     'If you want to use an already-trained model, please select Segment.'),
    ("Segment", lambda: show_frame(predict_page),
     'Select to run segmentation on your data.'),
    # ("Test", lambda: show_frame(track_page), "")
]

for i, (text, command, tooltip) in enumerate(buttons):
    btn = tk.Button(seg_page, text=text, width=15, command=command, font=('System', 16))
    btn.place(relx=0.5, rely=0.4 + i*0.15, anchor='center')
    CreateToolTip(btn, text=tooltip)

############
predictOutputPath = tk.StringVar()
epochs = 0
imageName = tk.StringVar()
trainModelName = tk.StringVar()
predictModelName = tk.StringVar()
gt_path = tk.StringVar()
val_path = tk.StringVar()
train_output_location = tk.StringVar()

contTrain = tk.IntVar()
transferLearn = tk.IntVar()
transferModel = tk.StringVar()
transferModelPath = tk.StringVar()

continue_training=0
transfer_learning=0

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


def callTrain(modl, epochs, gt_p, op_p, t_m, c_t, t_l,):
    print(modl, epochs, gt_p, op_p, t_m, c_t, t_l)
    segmentation.train(modl, epochs, gt_p, op_p, t_m, c_t, t_l)


def setPredictModelName(choice):
    predictModelName.set(choice)


def setTrainModelName(choice):
    trainModelName.set(choice)

def setWhetherContinueTraining():
    # continue_training = contTrain.get()
    print('/Continue Training') if contTrain.get() else print('\Training a New Model')

def setWhetherTransferLearning():
    # transfer_learning = transferLearn.get()
    print('/Using Transfer Learning') if transferLearn.get() else print('\Training from Scratch')


# def setWhetherTransferLearning(choice):
#     transferLearn.set(choice)
#     print('Using Transfer Learning') if transferLearn.get() else print('Training From Scratch')


def getAndSetPreTrainedModelForTL():
    transferModelPath.set(fd.askdirectory(initialdir=os.getcwd() + '/checkpoints/' + trainModelName.get()))
    transferModel.set(transferModelPath.get() + "/latest_model_" + "_" + transferModelPath.get().split('/')[-1] + '.ckpt')


##########################

modelNameList = ['FC-DenseNet', 'MobileUNet3D' ] # 'AdapNet3D', 'BiSeNet3D', 'DDSC3D', 'DeepLabp3D', 'DeepLabV3_plus3D', 'DenseASPP3D', 'Encoder_Decoder3D', 'FCN3D', 'FRRN3D', 'GCN3D', 'ICNet3D', 'PSPNet3D', 'RefineNet3D', 'resnet_utils3D', 'resnet_v23D']
numEpochs = tk.IntVar()

trainModelName.set('FC-DenseNet')

train_page_greet = tk.Label(train_page, text='Training', font=('Courier', 40, 'bold'))
train_page_greet.place(relx=0.5, rely=0.1, anchor='center')

# Back button (top-left)
button1 = tk.Button(train_page, text="Back", command=lambda: show_frame(seg_page), font=('System', 12))
button1.place(relx=0.05, rely=0.05, anchor='nw')

# Main form container
form_frame = tk.Frame(train_page)
form_frame.place(relx=0.5, rely=0.45, anchor='center')

# Model selection
tk.Label(form_frame, text="Model:", font=('System', 12)).grid(row=0, column=0, sticky='w', pady=5, padx=5)
modelMenu = tk.OptionMenu(form_frame, trainModelName, *modelNameList, command=setTrainModelName)
modelMenu.config(font=('System', 12), width=20)
modelMenu.grid(row=0, column=1, pady=5, sticky='w')

# Epochs
tk.Label(form_frame, text="No. of Epochs:", font=('System', 12)).grid(row=1, column=0, sticky='w', pady=5, padx=5)
entry1 = tk.Entry(form_frame, textvariable=numEpochs, font=('System', 12), width=15)
entry1.insert(0, '10')
entry1.grid(row=1, column=1, pady=5, sticky='w')

# Training data
button2 = tk.Button(form_frame, text="Training Data Folder", font=('System', 12), command=browseGT)
button2.grid(row=2, column=0, pady=5, padx=5, sticky='w')
entry2 = ttk.Entry(form_frame, textvariable=gt_path, width=25, font=('System', 12))
entry2.grid(row=2, column=1, pady=5, sticky='w')

# Output location
button3 = tk.Button(form_frame, text="Output Folder", font=('System', 12), command=trainOutputLocation)
button3.grid(row=3, column=0, pady=5, padx=5, sticky='w')
entry3 = ttk.Entry(form_frame, textvariable=train_output_location, width=25, font=('System', 12))
entry3.grid(row=3, column=1, pady=5, sticky='w')

# cboxes
checkbox_frame1 = tk.Frame(form_frame)
checkbox_frame1.grid(row=4, column=0, columnspan=3, pady=5, sticky='w')

tk.Label(checkbox_frame1, text='Continue Training', font=('System', 12)).pack(side='left', padx=(0, 10))

c1 = tk.Checkbutton(checkbox_frame1,
                   variable=contTrain,
                   command=setWhetherContinueTraining,
                   indicatoron=True,  # Keep as checkbox style
                   width=2,          # Makes the box wider
                   height=2,         # Makes the box taller
                   borderwidth=2,    # Thicker border
                   relief='ridge')   # More visible border
c1.pack(side='right')

checkbox_frame2 = tk.Frame(form_frame)
checkbox_frame2.grid(row=5, column=0, columnspan=3, pady=5, sticky='w')

tk.Label(checkbox_frame2, text='Transfer Learning', font=('System', 12)).pack(side='left', padx=(0, 10))

c2 = tk.Checkbutton(checkbox_frame2,
                   variable=transferLearn,
                   command=setWhetherTransferLearning,
                   indicatoron=True,
                   width=2,
                   height=2,
                   borderwidth=2,
                   relief='ridge')
c2.pack(side='right')

# Transfer learning model
buttonTxx4 = tk.Button(form_frame, text="Pre-Trained Model", font=('System', 12),
                      command=getAndSetPreTrainedModelForTL)
buttonTxx4.grid(row=6, column=0, pady=5, padx=5, sticky='w')
entryTxx4 = ttk.Entry(form_frame, textvariable=transferModel, width=25, font=('System', 12))
entryTxx4.grid(row=6, column=1, pady=5, sticky='w')

# Action buttons
button_frame = tk.Frame(train_page)
button_frame.place(relx=0.5, rely=0.85, anchor='center')

button4 = tk.Button(button_frame, text="Check", width=10, font=('System', 12),
                  command=lambda: print(trainModelName.get() + '\n' + str(numEpochs.get()) + '\n' + gt_path.get() + '\n' + train_output_location.get()))
button4.pack(side='left', padx=10)

button5 = tk.Button(button_frame, text="RUN", width=10, background="blue", foreground="white", font=('System', 12),
                  command=lambda: callTrain(trainModelName.get(), numEpochs.get(), gt_path.get(),
                                          train_output_location.get(), transferModel.get(),
                                          contTrain.get(), transferLearn.get()))
button5.pack(side='left', padx=10)

# Tooltips (maintained from original)
CreateToolTip(train_page_greet, text='You can use your own data with ground truth to train a segmentation model.\nIf you want to use an already-trained model, please go back and select Predict.')
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
predictModelName.set('FC-DenseNet')

def getTrainedModel():
    trainedModelPath.set(fd.askdirectory(initialdir=os.getcwd() + '/checkpoints/' + predictModelName.get()))
    trainedModel.set(trainedModelPath.get() + "/latest_model_" + "_" + trainedModelPath.get().split('/')[-1] + '.ckpt')


def callPredict(model, image, startTime, endTime, trModelPath, predictOPpath):
    segmentation_predict.predict(model=model, image=image, startpoint=int(startTime), endpoint=int(endTime),
                                 modelCheckpointName=trModelPath, op_folder=predictOPpath)

startT = tk.StringVar()
endT = tk.StringVar()


predict_page_greet = tk.Label(predict_page, text='Inference', font=('Courier', 40, 'bold'))
predict_page_greet.place(relx=0.5, rely=0.1, anchor='center')

# Back button
buttonPr1 = tk.Button(predict_page, text="Back", command=lambda: show_frame(seg_page), font=('System', 12))
buttonPr1.place(relx=0.05, rely=0.05, anchor='nw')

# Main form container
form_frame = tk.Frame(predict_page)
form_frame.place(relx=0.5, rely=0.45, anchor='center')

# Model selection
tk.Label(form_frame, text="Model  ", font=('System', 12)).grid(row=0, column=0, sticky='w', pady=5, padx=5)
modelMenu = tk.OptionMenu(form_frame, predictModelName, *modelNameList, command=setPredictModelName)
modelMenu.config(font=('System', 12), width=20)
modelMenu.grid(row=0, column=1, pady=5, sticky='w')

# Time range
tk.Label(form_frame, text="Start Time  ", font=('System', 12)).grid(row=1, column=0, sticky='w', pady=5, padx=5)
entryPr1 = tk.Entry(form_frame, textvariable=startT, font=('System', 12), width=10)
entryPr1.insert(0, '1')
entryPr1.grid(row=1, column=1, pady=5, sticky='w')

tk.Label(form_frame, text="End Time  ", font=('System', 12)).grid(row=2, column=0, sticky='w', pady=5, padx=5)
entryPr2 = tk.Entry(form_frame, textvariable=endT, font=('System', 12), width=10)
entryPr2.insert(0, '1')
entryPr2.grid(row=2, column=1, pady=5, sticky='w')

# Image selection
buttonPr2 = tk.Button(form_frame, text="Input Image", font=('System', 12), command=browseImage)
buttonPr2.grid(row=3, column=0, pady=5, padx=5, sticky='w')
entryPr3 = tk.Entry(form_frame, textvariable=imageName, font=('System', 12))
entryPr3.grid(row=3, column=1, pady=5, sticky='w')

# Output location
buttonPr3 = tk.Button(form_frame, text="Output Folder", font=('System', 12), command=predictOutputLocation)
buttonPr3.grid(row=4, column=0, pady=5, padx=5, sticky='w')
entryPr4 = tk.Entry(form_frame, textvariable=predictOutputPath, font=('System', 12))
entryPr4.grid(row=4, column=1, pady=5, sticky='w')

# Model path
buttonPr4 = tk.Button(form_frame, text="Trained Model", font=('System', 12), command=getTrainedModel)
buttonPr4.grid(row=5, column=0, pady=5, padx=5, sticky='e')
entryPr5 = tk.Entry(form_frame, textvariable=trainedModel, font=('System', 12))
entryPr5.grid(row=5, column=1, pady=5, sticky='w')

# Action buttons
button_frame = tk.Frame(predict_page)
button_frame.place(relx=0.5, rely=0.85, anchor='center')

buttonPr5 = tk.Button(button_frame, text="Check", width=10, font=('System', 12),
                    command=lambda: print(predictModelName.get(), imageName.get(),
                                        startT.get(), endT.get(), predictOutputPath.get()))
buttonPr5.pack(side='left', padx=10)

buttonPr6 = tk.Button(button_frame, text="RUN", width=10, background="blue", foreground="white", font=('System', 12),
                    command=lambda: callPredict(predictModelName.get(), imageName.get(),
                                              startT.get(), endT.get(), trainedModel.get(),
                                              predictOutputPath.get()))
buttonPr6.pack(side='left', padx=10)

# Tooltips (maintained from original)
CreateToolTip(buttonPr2, text='Select the Image you want to run segmentation on.')
CreateToolTip(buttonPr3, text='Select the folder to save the outputs of segmentation.')
CreateToolTip(buttonPr5, text='Click to check the console if everything you selected is correct.')
CreateToolTip(buttonPr6, text='Start segmentation of your image!')

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

# Title
track_page_greet = tk.Label(track_page, text='Tracking', font=('Courier', 40, 'bold'))
track_page_greet.place(relx=0.5, rely=0.1, anchor='center')

# Back button
buttonTr1 = tk.Button(track_page, text="Back", command=lambda: show_frame(homepage), font=('System', 12))
buttonTr1.place(relx=0.05, rely=0.05, anchor='nw')

# Main form frame
form_frame = tk.Frame(track_page)
form_frame.place(relx=0.5, rely=0.45, anchor='center')

# Row 0: Original Images
buttonTr2 = tk.Button(form_frame, text="Select Original Images", font=('System', 12), command=browseImageTr)
buttonTr2.grid(row=0, column=0, padx=5, pady=5, sticky='w')

entryTr1 = ttk.Entry(form_frame, textvariable=imgname, font=('System', 12))
entryTr1.grid(row=0, column=1, padx=5, pady=5, sticky='w')
# Row 1: Segmentation Results
buttonTr3 = tk.Button(form_frame, text="Select Folder with Segmentation Results", font=('System', 12), command=segOPfolder)
buttonTr3.grid(row=1, column=0, padx=5, pady=5, sticky='w')

entryTr2 = ttk.Entry(form_frame, textvariable=segloc, font=('System', 12))
entryTr2.grid(row=1, column=1, padx=5, pady=5, sticky='w')
# Row 2: Tracking Output Folder
buttonTr4 = tk.Button(form_frame, text="Select Folder to Save Tracking Results", font=('System', 12), command=trackOPfolder)
buttonTr4.grid(
    row=2, column=0, padx=5, pady=5, sticky='w')
entryTr3 = ttk.Entry(form_frame, textvariable=trackloc, font=('System', 12))
entryTr3.grid(row=2, column=1, padx=5, pady=5, sticky='w')
# Row 3: Start Time
labelTr1 = tk.Label(form_frame, text="Start Time", font=('System', 12))
labelTr1.grid(row=3, column=0, padx=5, pady=5, sticky='w')
entryTr4 = ttk.Entry(form_frame, textvariable=strT, font=('System', 12), width=10)
entryTr4.insert(0, '1')
entryTr4.grid(row=3, column=1, padx=5, pady=5, sticky='w')
# Row 4: End Time
labelTr2 = tk.Label(form_frame, text="Total Frames to Track", font=('System', 12))
labelTr2.grid(row=4, column=0, padx=5, pady=5, sticky='w')
entryTr5 = ttk.Entry(form_frame, textvariable=enT, font=('System', 12), width=10)
entryTr5.insert(0, '41')
entryTr5.grid(row=4, column=1, padx=5, pady=5, sticky='w')
# Row 5: Trackback Time
labelTr3 = tk.Label(form_frame, text="Trackback Time", font=('System', 12))
labelTr3.grid(row=5, column=0, padx=5, pady=5, sticky='w')
entryTr6 = ttk.Entry(form_frame, textvariable=trbT, font=('System', 12), width=10)
entryTr6.insert(0, '2')
entryTr6.grid(row=5, column=1, padx=5, pady=5, sticky='w')
# Row 6: Min Object Size
labelTr4 = tk.Label(form_frame, text="Min Size Threshold", font=('System', 12))
labelTr4.grid(row=6, column=0, padx=5, pady=5, sticky='w')
entryTr7 = ttk.Entry(form_frame, textvariable=ost, font=('System', 12), width=10)
entryTr7.insert(0, '6')
entryTr7.grid(row=6, column=1, padx=5, pady=5, sticky='w')
# Bottom Button Frame
button_frame = tk.Frame(track_page)
button_frame.place(relx=0.5, rely=0.85, anchor='center')

buttonTr5 = tk.Button(button_frame, width=10, text="Check", font=('System', 12),
                      command=lambda: print("Check check check"))
buttonTr5.pack(side='left', padx=10)

buttonTr6 = tk.Button(button_frame, width=10, text="RUN", font=('System', 12),
                      background="blue", foreground="white",
                      command=lambda: runTracking(
                          imageName=imgname.get(),
                          segmentationOPFolder=segloc.get(),
                          startTime=int(strT.get()),
                          endTime=int(enT.get()),
                          trackbackTime=int(trbT.get()),
                          min_obj_size=int(ost.get()),
                          protein1Name=p1n.get(),
                          protein2Name=p2n.get()
                      ))
buttonTr6.pack(side='left', padx=10)


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

# def generateFamilyTree():
#     print('Generating Family Tree for your object...')

# Page title
fam_page_greet = tk.Label(fam_page, text='Family Tree', font=('Courier', 40, 'bold'))
fam_page_greet.place(relx=0.5, rely=0.1, anchor='center')

# Back button
buttonFt1 = tk.Button(fam_page, text="Back", command=lambda: show_frame(homepage), font=('System', 12))
buttonFt1.place(relx=0.05, rely=0.05, anchor='nw')

# Form Frame
form_frame = tk.Frame(fam_page)
form_frame.place(relx=0.5, rely=0.45, anchor='center')

# Row 0: Select Tracking Results
buttonFt2 = tk.Button(form_frame, text="Select Tracking Results", font=('System', 12), command=selectCsvFile)
buttonFt2.grid(row=0, column=0, padx=5, pady=5, sticky='w')
entryFt2 = tk.Entry(form_frame, textvariable=excelFile, font=('System', 12), width=20)
entryFt2.grid(row=0, column=1, padx=5, pady=5, sticky='w')

# Row 1: Select Output Folder
buttonFt3 = tk.Button(form_frame, text="Folder to Save", font=('System', 12), command=selectFtOutputFolder)
buttonFt3.grid(row=1, column=0, padx=5, pady=5, sticky='w')
entryFt3 = tk.Entry(form_frame, textvariable=ftOpFolder, font=('System', 12), width=20)
entryFt3.grid(row=1, column=1, padx=5, pady=5, sticky='w')

# Row 2: Branch Length Threshold
buttonFt4 = tk.Button(form_frame, text='Branch Length Threshold', font=('System', 12))
buttonFt4.grid(row=2, column=0, padx=5, pady=5, sticky='w')
entryFt4 = tk.Entry(form_frame, textvariable=branchMinLen, font=('System', 12), width=20)
entryFt4.insert(0, '1')
entryFt4.grid(row=2, column=1, padx=5, pady=5, sticky='w')

# Row 3: Object ID
buttonFt8 = tk.Button(form_frame, text='Enter Object ID', font=('System', 12))
buttonFt8.grid(row=3, column=0, padx=5, pady=5, sticky='w')
entryFt5 = tk.Entry(form_frame, textvariable=objId, font=('System', 12), width=20)
entryFt5.grid(row=3, column=1, padx=5, pady=5, sticky='w')

# Row 4: Label for valid input format
labelFTB6 = tk.Label(form_frame, text="Valid Inputs:  | 76 | 1,2,13 | 1-29 | 1-4,23-27,88-121 | all |", font=('System', 10))
labelFTB6.grid(row=4, column=1, columnspan=2, pady=(10, 5), sticky='w')

# Action Buttons Frame
button_frame = tk.Frame(fam_page)
button_frame.place(relx=0.5, rely=0.82, anchor='center')

# Button: Generate for IDs
buttonFt6 = tk.Button(button_frame, text="Generate Family Tree for Entered ID(s)", font=('System', 12),
                      background='blue', foreground='white',
                      command=lambda: generateFamilyTree(
                          excelFile=excelFile.get(),
                          ftFolder=ftOpFolder.get(),
                          tidlist=objId.get()))
buttonFt6.pack(side='left', padx=10)

# Optional: Button to generate all (uncomment if needed)
# buttonFt7 = tk.Button(button_frame, text="Generate Family Trees for All Target IDs", font=('System', 12),
#                       background='navy', foreground='white',
#                       command=lambda: generateFamilyTrees(
#                           excelFile=excelFile.get(),
#                           ftFolder=ftOpFolder.get()))
# buttonFt7.pack(side='left', padx=10)

# Tooltips
CreateToolTip(buttonFt2, text='Select the CSV file that contains the tracking data.\nYou can find it in the tracking results folder.')
CreateToolTip(buttonFt3, text='Click here to select a folder to store your family trees.\nPlease select an empty folder.')
CreateToolTip(buttonFt4, text='Set the minimum number of frames an object must exist\nto be considered in the family tree.')
# CreateToolTip(buttonFt7, text='Click here to draw family trees for ALL target IDs! \n Target IDs are object IDs that start as new objects, not a result of split event. '
#                               '\n Use "Generate Family Tree" button with "all" as object ID to generate family tree for every single object.')
# CreateToolTip(buttonFt6, text='Click here to draw family trees for the entered IDs! \n' \
#  "Follow one of the following Format:\n" \
#                               "\tValid input: 12\n" \
#                "\tValid input: 1,2,13,62\n" \
#                "\tValid Input: 76\n" \
#                "\tValid Input: 1-29\n" \
#                "\tValid Input: 1-4,23-27, 88-121\n"\
#                 "\tValid Input: 1-4,19, 88-121\n" \
#               "\tall (generate for all IDs, may take longer.")
########################################################################################################################


# # Analysis
# # corss corr
# ftOpFolderCC = tk.StringVar()
# segOPfolderCC = tk.StringVar()
# excelFileCC = tk.StringVar()


# def selectSegmentationFolderCC():
#     None


# def selectExcelFileCC():
#     None


# def selectFTFolderCC():
#     None


# cross_corr_greet = tk.Label(cross_corr_page, text='Cross-Correlation Analysis', font=('Courier', 40, 'bold'))
# cross_corr_greet.place(x=50, y=120)

# buttonCC1 = tk.Button(cross_corr_page, text="Back", command=lambda: show_frame(homepage), font=('System', 12))

# buttonCC2 = tk.Button(cross_corr_page, text="Select Segmentation Results Folder",
#                       command=lambda: selectSegmentationFolderCC(), font=('System', 12))
# entryCC2 = tk.Entry(cross_corr_page, textvariable=segOPfolderCC, font=('System', 12))
# buttonCC3 = tk.Button(cross_corr_page, text="Select Tracking Results", command=lambda: selectExcelFileCC(),
#                       font=('System', 12))

# entryCC3 = tk.Entry(cross_corr_page, textvariable=excelFileCC, font=('System', 12))
# buttonCC4 = tk.Button(cross_corr_page, text='Select Family Tree Folder', font=('System', 12),
#                       command=lambda: selectFTFolderCC())
# entryCC4 = tk.Entry(cross_corr_page, textvariable=ftOpFolderCC, font=('System', 12))
# optionCC = tk.Label(cross_corr_page, borderwidth=1, relief='solid', text='Select Options to Run', font=('System', 12))
# c1 = tk.Checkbutton(cross_corr_page, text='Cross Correlation Calculation', onvalue=1, font=('System', 12))
# c2 = tk.Checkbutton(cross_corr_page, text='Statistical Analysis', onvalue=1, font=('System', 12))

# buttonCC6 = tk.Button(cross_corr_page, width=10, text='RUN', font=('System', 12), background='blue', foreground='white')

# buttonCC1.place(x=50, y=50)
# buttonCC2.place(x=50, y=200);
# entryCC2.place(x=500, y=200)
# buttonCC3.place(x=50, y=250);
# entryCC3.place(x=500, y=250)
# buttonCC4.place(x=50, y=300);
# entryCC4.place(x=500, y=300)
# optionCC.place(x=50, y=350)
# c1.place(x=70, y=400)
# c2.place(x=70, y=450)
# buttonCC6.place(x=50, y=550)

################ ANALYSIS ###########################################################################################

origImageA = tk.StringVar()
segFolderA = tk.StringVar()
trackFolderA = tk.StringVar()
sT = tk.StringVar()
at1 = tk.StringVar()
at2 = tk.StringVar()



def setSegFolder():
    segFolderA.set(fd.askdirectory())
def setTrackFolder():
    trackFolderA.set(fd.askdirectory())
def browseOrigImage():
    origImageA.set(fd.askopenfilename(defaultextension='.tif', filetypes=[("TIF Files", "*.tif"),
                                                                       ("TIFF Files", "*.tiff"),
                                                                       ("NIFTI Files", "*.nii")]))

# Title
analysis_page_greet = tk.Label(analysis_page, text='Analysis', font=('Courier', 40, 'bold'))
analysis_page_greet.place(relx=0.5, rely=0.1, anchor='center')

# Back button
buttonA1 = tk.Button(analysis_page, text="Back", command=lambda: show_frame(homepage), font=('System', 12))
buttonA1.place(relx=0.05, rely=0.05, anchor='nw')

# Form Frame
form_frame = tk.Frame(analysis_page)
form_frame.place(relx=0.5, rely=0.45, anchor='center')

# Row 0: Original Image
buttonA2 = tk.Button(form_frame, text="Select Original Images", font=('System', 12), command=browseOrigImage)
buttonA2.grid(row=0, column=0, padx=5, pady=5, sticky='w')
entryA2 = ttk.Entry(form_frame, textvariable=origImageA, font=('System', 10), width=20)
entryA2.grid(row=0, column=1, padx=5, pady=5, sticky='w')

# Row 1: Segmentation Folder
buttonA3 = tk.Button(form_frame, text="Select Folder with Segmentation Results", font=('System', 12), command=setSegFolder)
buttonA3.grid(row=1, column=0, padx=5, pady=5, sticky='w')
entryA3 = ttk.Entry(form_frame, textvariable=segFolderA, font=('System', 10), width=20)
entryA3.grid(row=1, column=1, padx=5, pady=5, sticky='w')

# Row 2: Tracking Folder
buttonA4 = tk.Button(form_frame, text="Select Folder with Tracking Results", font=('System', 12), command=setTrackFolder)
buttonA4.grid(row=2, column=0, padx=5, pady=5, sticky='w')
entryA4 = ttk.Entry(form_frame, textvariable=trackFolderA, font=('System', 10), width=20)
entryA4.grid(row=2, column=1, padx=5, pady=5, sticky='w')

# Row 3: Start Time
labelA1 = tk.Label(form_frame, text='Start Time', font=('System', 12))
labelA1.grid(row=3, column=0, padx=5, pady=5, sticky='w')
entryAL1 = ttk.Entry(form_frame, textvariable=at1, font=('System', 12), width=10)
entryAL1.insert(0, '1')
entryAL1.grid(row=3, column=1, padx=5, pady=5, sticky='w')

# Row 4: End Time
labelA2 = tk.Label(form_frame, text='End Time', font=('System', 12))
labelA2.grid(row=4, column=0, padx=5, pady=5, sticky='w')
entryAL2 = ttk.Entry(form_frame, textvariable=at2, font=('System', 12), width=10)
entryAL2.insert(0, '41')
entryAL2.grid(row=4, column=1, padx=5, pady=5, sticky='w')

# Button Frame
button_frame = tk.Frame(analysis_page)
button_frame.place(relx=0.5, rely=0.82, anchor='center')

buttonA5 = tk.Button(button_frame, width=10, text="RUN", font=('System', 12),
                     background="blue", foreground="white",
                     command=lambda: all_analysis_app(
                         trackedimagepath=trackFolderA.get() + '/TrackedCombined.nii',
                         segPath=segFolderA.get(),
                         origImgPath=origImageA.get(),
                         t1=int(at1.get()),
                         t2=int(at2.get())
                     ))
buttonA5.pack()


########################################################################################################################





########################################################################################################################

help_page_greet = tk.Label(help_page, text='Help', font=('Courier', 40, 'bold'))
help_page_greet.place(x=50, y=120)

# ----------------------------------------------------------------------------------------------------------------------

filesFolder = tk.StringVar()
allFiles = tk.StringVar()


def selectFilesFolder():
    filesFolder.set(fd.askdirectory())


convert_page_greet = tk.Label(convert_page, text='Convert .czi and .tif to .nii', font=('Courier', 40, 'bold'))
buttonCon1 = tk.Button(convert_page, text="Back", command=lambda: show_frame(homepage), font=('System', 12))
buttonCon2 = tk.Button(convert_page, text="Select Folder with Files", command=lambda: selectFilesFolder(),
                       font=('System', 12))
entryCon1 = tk.Entry(convert_page, textvariable=filesFolder, font=('System', 12))

buttonCon3 = tk.Button(convert_page, text="Convert .czi to .nii", command=lambda: czi2nii(filesFolder.get()),
                       font=('System', 12))
buttonCon4 = tk.Button(convert_page, text="Convert .tif to .nii", command=lambda: tif2nii(filesFolder.get()),
                       font=('System', 12))

buttonCon1.place(x=50, y=50)
buttonCon2.place(x=50, y=100)
entryCon1.place(x=500, y=100)
buttonCon3.place(x=50, y=250)
buttonCon4.place(x=50, y=300)

# ----------------------------------------------------------------------------------------------------------------------

window.mainloop()
