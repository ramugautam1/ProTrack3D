from __future__ import print_function

import gc

from segment.load_data_nii import loadDataGeneral as loadDataGeneral_nii
from segment.load_data_nii import loadDataGeneral_test as loadDataGeneral_test_nii
from segment.load_data import loadDataGeneral
from segment.build_model import build_mymodel
import segment.utils as utils
import segment.FCN as FCN
# import FCN2, models.MobileUNet3D, models.resnet_v23D, models.Encoder_Decoder3D, models.Encoder_Decoder3D_contrib, models.DeepLabp3D, models.DeepLabV33D
# import models.FRRN3D, models.FCN3D, models.GCN3D, models.AdapNet3D, models.ICNet3D, models.PSPNet3D, models.RefineNet3D, models.BiSeNet3D, models.DDSC3D
# import models.DenseASPP3D, models.DeepLabV3_plus3D

import matplotlib.pyplot as plt
import numpy as np # Path to csv-file. File should contain X-ray filenames as first column,
        # mask filenames as second column.
import nibabel as nib
# from keras.models import load_model
import math as math
from skimage.color import hsv2rgb, rgb2hsv, gray2rgb
from skimage import io, exposure


from skimage.color import hsv2rgb, rgb2hsv, gray2rgb
from skimage import io, exposure
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os,time,cv2, sys, math
import numpy as np
import cv2, glob
import time, datetime
import argparse
import random
import os, sys
import subprocess
import segment.helpers as helpers
import pandas as pd
import segment.createTrainingData as createTrainingData

def IoU(y_true, y_pred):
    assert y_true.dtype == bool and y_pred.dtype == bool
    print(y_true.shape, y_pred.shape)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)


def Dice(y_true, y_pred):
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator


def balanced_cross_entropy(beta, y_true, y_pred):
    weight_a = beta * tf.cast(y_true, tf.float32)
    weight_b = (1 - beta) * tf.cast(1 - y_true, tf.float32)

    o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b
    return tf.reduce_mean(o)


def focal_loss(y_true, logits, alpha=0.25, gamma=2):
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        targets = tf.cast(targets, tf.float32)
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (
                    weight_a + weight_b) + logits * weight_b

    y_pred = tf.math.sigmoid(logits)
    loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

    return tf.reduce_mean(loss)


def saggital(img):
    """Extracts midle layer in saggital axis and rotates it appropriately."""
    return img[:,  int(img.shape[1] / 2), ::-1].T


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def train(model,epochs,gt_path,op_path):
    dataset_path,valdataset_path = createTrainingData.create(gt_path)
  
    dataset = os.path.basename(dataset_path)
    valdataset = os.path.basename(valdataset_path)



    num_epochs = epochs
    model = model
    mode = 'train'
    op_path = op_path

    ckpt_path = os.getcwd() + "/checkpoints/" + model + '/' + dataset+num_epochs

    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_balancing', type=str2bool, default=True, help='Whether to use median frequency class weights to balance the classes in the loss')
    parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
    parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of images in each batch')
    parser.add_argument('--class_weight_reference', type=str, default="reference/Ecad2020", help='reference you are using.')
    parser.add_argument('--num_val_images', type=int, default=200, help='Number of ramdom validation samples')
    
    
    """
    Currently, FC-DenseNet is the best model.
    PSPNet must take input size 192 for 3D
    """
    """
    Try to accommodate the input size.
    Input size for Ecad2017: 128x128x13
    Input size for Ecad2020: 32x32x15
    Input size for Aju2020: 32x32x15
    
    Output size for Ecad2017: 32x35x13
    Output size for Ecad2020: 35x32x15
    Output size for Aju2020: 35x32x15
    """
    args = parser.parse_args()
    
    img_size = 32
    # class_names_list, label_values = helpers.get_label_info(os.path.join(dataset_path, "class_dict.csv"))
    class_names_list, label_values = helpers.get_label_info(dataset_path + "/class_dict.csv")
    class_names_string = ""
    for class_name in class_names_list:
        if not class_name == class_names_list[-1]:
            class_names_string = class_names_string + class_name + ", "
        else:
            class_names_string = class_names_string + class_name
    
    num_classes = len(label_values)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess=tf.Session(config=config)
    
    net_input = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 13, 1])
    net_output = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 13, num_classes])
    
    network = None
    init_fn = None
    print(model)
    if model == "mymodel":
        network = build_mymodel(net_input)
    elif model == "FC-DenseNet":
        network = FCN.build_fc_densenet(net_input)
    '''
    # elif model == "MobileUNet3D-Skip":
    #     network = models.MobileUNet3D.build_mobile_unet3D(net_input, 'MobileUNet3D-Skip', 2)
    # elif model == "FC-DenseNet103":
    #     network = FCN2.build_fc_densenet(net_input,num_classes=num_classes)
    # elif model == "ResNet-101":
    #     network = models.resnet_v23D.resnet_v2_101(net_input, num_classes=num_classes)
    # elif model == "Encoder_Decoder3D":
    #     network = models.Encoder_Decoder3D.build_encoder_decoder(net_input, num_classes=num_classes)
    #     # RefineNet requires pre-trained ResNet weights
    # elif model == "Encoder_Decoder3D_contrib":
    #     network = models.Encoder_Decoder3D_contrib.build_encoder_decoder(net_input, num_classes=num_classes)
    # elif model == "DeepLabV3p3D":
    #     network = models.DeepLabp3D.Deeplabv3(net_input, num_classes)
    # elif model == "DeepLabV33D-Res50" or model == "DeepLabV33D-Res101" or model == "DeepLabV33D-Res152":
    #     network = models.DeepLabV33D.build_deeplabv3(net_input, num_classes=num_classes, preset_model=model)
    # elif model == "DeepLabV3_plus-Res50" or model == "DeepLabV3_plus-Res101" or model == "DeepLabV3_plus-Res152":
    #     network = models.DeepLabV3_plus3D. build_deeplabv3_plus(net_input, num_classes=num_classes, preset_model=model)
    # elif model == "FRRN-A" or model == "FRRN-B":
    #     network = models.FRRN3D.build_frrn(net_input, num_classes=num_classes, preset_model=model)
    # elif model == "FCN8":
    #     network = models.FCN3D.build_fcn8(net_input, num_classes=num_classes)
    #     # RefineNet requires pre-trained ResNet weights
    # elif model == "GCN-Res50" or model == "GCN-Res101" or model == "GCN-Res152":
    #     network = models.GCN3D.build_gcn(net_input, num_classes=num_classes, preset_model=model)
    # elif model == "AdapNet3D":
    #     network = models.AdapNet3D.build_adaptnet(net_input, num_classes=num_classes)
    # elif model == "ICNet-Res50" or model == "ICNet-Res101" or model == "ICNet-Res152":
    #     network = models.ICNet3D.build_icnet(net_input, [img_size, img_size, 13], num_classes=num_classes, preset_model=model)
    # elif model == "PSPNet-Res50" or model == "PSPNet-Res101" or model == "PSPNet-Res152":
    #     network = models.PSPNet3D.build_pspnet(net_input, [img_size, img_size, 13], num_classes=num_classes, preset_model=model)
    # elif model == "RefineNet-Res50" or model == "RefineNet-Res101" or model == "RefineNet-Res152":
    #     network = models.RefineNet3D.build_refinenet(net_input, num_classes=num_classes, preset_model=model)
    # elif model == "BiSeNet-ResNet50" or model == "BiSeNet-Res101" or model == "BiSeNet-Res152":
    #     network = models.BiSeNet3D.build_bisenet(net_input, num_classes=num_classes, preset_model=model)
    # elif model == "DDSC-ResNet50" or model == "DDSC-Res101" or model == "DDSC-Res152":
    #     network = models.DDSC3D.build_ddsc(net_input, num_classes=num_classes, preset_model=model)
    # elif model == "DenseASPP-ResNet50" or model == "DenseASPP-Res101" or model == "DenseASPP-Res152":
    #     network = models.DenseASPP3D.build_dense_aspp(net_input, num_classes=num_classes, preset_model=model)
    '''
#---------------------------------------------------------------------------------------------------------------------------------------------

    losses = None
    if args.class_balancing:
        print("Computing class weights for", dataset, "...")
        class_weights = utils.compute_class_weights(labels_dir=args.class_weight_reference,
                                                    label_values=label_values)
        print(class_weights)
        print(net_output,network[0])
        weights = tf.reduce_sum(class_weights * net_output, axis=-1)
        unweighted_loss = None
        unweighted_loss = 100 * tf.nn.softmax_cross_entropy_with_logits(logits=network[0], labels=net_output) #+ 10 * focal_loss(logits=network[0], y_true=net_output)
        # unweighted_loss = dice_loss(net_output, network[0])
        # unweighted_loss = focal_loss(logits=network[0], y_true=net_output)
        losses = unweighted_loss * weights

        # losses = balanced_cross_entropy(beta=weights[0], y_true=net_output, y_pred=network[0])

    else:
        # losses = tf.nn.softmax_cross_entropy_with_logits(logits=network[0], labels=net_output)
        losses = dice_loss(net_output, network[0])
    loss = tf.reduce_mean(losses)

    opt = tf.train.AdamOptimizer(0.0002).minimize(loss, var_list=[var for var in tf.trainable_variables()])
    
    
    
    
    saver = tf.train.Saver(max_to_keep=1000)
    sess.run(tf.global_variables_initializer())

    if init_fn is not None:
        init_fn(sess)


    model_checkpoint_name = ckpt_path + "/latest_model_" + "_" + dataset + ".ckpt"


    if args.continue_training or not mode == "train":
        print('Loaded latest model checkpoint')
        print(model_checkpoint_name)
        saver.restore(sess, model_checkpoint_name)

    # Load the data
    print("Loading the data ...")
    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    csv_path_training = dataset_path + '/idx_train.csv'
    # Path to the folder with images. Images will be read from path + path_from_csv
    path1 = csv_path_training[:csv_path_training.rfind('/')] + '/'

    df = pd.read_csv(csv_path_training)

    # Load test data
    X_train, y_train = loadDataGeneral(df, path1, img_size)
    print(X_train.shape, y_train.shape)

    csv_path_val = valdataset_path + '/idx_val.csv'
    # Path to the folder with images. Images will be read from path + path_from_csv
    path2 = csv_path_val[:csv_path_val.rfind('/')] + '/'

    df = pd.read_csv(csv_path_val)

    # Load val data
    X_val, y_val = loadDataGeneral(df, path2, img_size)


    # n_val = X_val.shape[0]
    # inpShape_val = X_val.shape[1:]
    # print(X_val.shape, y_val.shape)
    print('length of training sample:', len(X_train))
    avg_loss_per_epoch = []
    val_indices = []
    num_vals = min(args.num_val_images, len(X_val))
    #num_vals = len(X_val)
    random.seed(16)
    val_indices = random.sample(range(0, len(X_val)), num_vals)

    avg_scores_per_epoch = []

    # Do the training here
    avg_iou_list=[]
    for epoch in range(0, num_epochs):
        current_losses = []

        cnt = 0

        # Equivalent to shuffling
        num_iters = int(np.floor(len(X_train) / args.batch_size))
        st = time.time()
        epoch_st = time.time()
        for i in range(num_iters):
            # st=time.time()

            input_image_batch = []
            output_image_batch = []

            # Collect a batch of images
            # Collect a batch of images
            for j in range(args.batch_size):
                index = i * args.batch_size + j
                input_image = X_train[index]
                output_image = y_train[index]
                with tf.device('/cpu:0'):
                    #input_image, output_image = data_augmentation(input_image, output_image)
                    # Prep the data. Make sure the labels are in one-hot format
                    input_image = np.float32(input_image) #/ 255.0
                    output_image = np.float32(helpers.one_hot_it(label=output_image, label_values=label_values))

                    # input_image_batch.append(np.expand_dims(input_image, axis=0))
                    # output_image_batch.append(np.expand_dims(output_image, axis=0))
                    input_image_batch.append(input_image)
                    output_image_batch.append(output_image)


            #print(type(input_image_batch[0].shape))
            if args.batch_size == 1:
                input_image_batch = np.asarray(input_image_batch[0])
                output_image_batch = np.asarray(output_image_batch[0])
            else:
                # input_image_batch = np.squeeze(np.stack(input_image_batch, axis=0))
                # output_image_batch = np.squeeze(np.stack(output_image_batch, axis=0))
                input_image_batch = np.asarray(np.stack(input_image_batch, axis=0))
                output_image_batch = np.asarray(np.stack(output_image_batch, axis=0))
            #print(type(input_image_batch), input_image_batch.shape)

            # Do the training
            _, current= sess.run([opt, loss],
                            feed_dict={net_input: input_image_batch, net_output: output_image_batch})
            #_, current, stack = sess.run([opt, loss], feed_dict={net_input: input_image_batch, net_output: output_image_batch})
            current_losses.append(current)
            cnt = cnt + args.batch_size
            if cnt % 100 == 0:
                string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f" % (
                epoch, cnt, current, time.time() - st)
                utils.LOG(string_print)
                st = time.time()

        mean_loss = np.mean(current_losses)
        avg_loss_per_epoch.append(mean_loss)

        # Create directories if needed
        if not os.path.isdir("%s/%04d" % (ckpt_path, epoch)):
            os.makedirs("%s/%04d" % (ckpt_path,  epoch))

        # Save latest checkpoint to same file name
        print("Saving latest checkpoint")
        saver.save(sess, model_checkpoint_name)

        # if epoch % args.checkpoint_step == 0:
        #     print("Saving checkpoint for this epoch")
        #     saver.save(sess, "%s/%s/%s/%04d/model.ckpt" % ("checkpoints",model, dataset, epoch))
        

        if epoch % args.validation_step == 0:
            print("Performing validation")
            target = open("%s/%04d/val_scores.csv" % (ckpt_path, epoch), 'w')
            target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))

            scores_list = []
            class_scores_list = []
            precision_list = []
            recall_list = []
            f1_list = []
            iou_list = []

            ious = np.zeros(len(y_val))
            dices = np.zeros(len(y_val))
            ious_back = np.zeros(len(y_val))
            dices_back = np.zeros(len(y_val))

            # Do the validation on a small set of validation images
            for ind in val_indices:
                # input_image = np.expand_dims(
                #     np.float32(X_val[ind]), axis=0) #/255.0
                input_image = np.expand_dims(
                   X_val[ind], axis=0)  # /255.0
                gt = y_val[ind]
                gt = np.array(gt)

                gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))
                #gt = 1-gt

                #output_image, stack= sess.run(network, feed_dict={net_input: input_image})
                output_image, stack = sess.run(network, feed_dict={net_input: input_image})

                output_image = np.array(output_image[0, :, :, :, :])
                # print(output_image.shape)
                output_image = helpers.reverse_one_hot(output_image)
                #output_image = 1-output_image
                #print(output_image)

                # fig = plt.figure(figsize=(16, 16))
                # columns = 13
                # rows = 1
                # for i in range(1, rows * columns + 1):
                #     img = gt[:, :, i - 1]
                #     img=1-img
                #     fig.add_subplot(1, columns, i)
                #     plt.imshow(img)
                # fig.savefig("%s/%s/%04d/%s_gt.png"%("checkpoints",model, epoch, str(ind)))
                # plt.close(fig)
                #
                # fig = plt.figure(figsize=(16, 16))
                # columns = 13
                # rows = 1
                # for i in range(1, rows * columns + 1):
                #     img = output_image[:, :, i - 1]
                #     img = 1 - img
                #     fig.add_subplot(1, columns, i)
                #     plt.imshow(img)
                # fig.savefig("%s/%s/%04d/%s_pred.png" % ("checkpoints", model ,epoch, str(ind)))
                # plt.close(fig)

                accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image,
                                                                                         label=gt,
                                                                                         num_classes=num_classes)
                target.write("%s, %f, %f, %f, %f, %f" % ('val', accuracy, prec, rec, f1, iou))
                for item in class_accuracies:
                    target.write(", %f"%(item))
                target.write("\n")

                scores_list.append(accuracy)
                class_scores_list.append(class_accuracies)
                precision_list.append(prec)
                recall_list.append(rec)
                f1_list.append(f1)
                iou_list.append(iou)

            target.close()

            avg_score = np.mean(scores_list)
            class_avg_scores = np.mean(class_scores_list, axis=0)
            avg_scores_per_epoch.append(avg_score)
            avg_precision = np.mean(precision_list)
            avg_recall = np.mean(recall_list)
            avg_f1 = np.mean(f1_list)
            avg_iou = np.mean(iou_list)
            avg_iou_list.append(avg_iou)

            print("\nAverage validation accuracy for epoch # %04d = %f" % (epoch, avg_score))
            print("Average per class validation accuracies for epoch # %04d:" % (epoch))
            for index, item in enumerate(class_avg_scores):
                print("%s = %f" % (class_names_list[index], item))
            print("Validation precision = ", avg_precision)
            print("Validation recall = ", avg_recall)
            print("Validation F1 score = ", avg_f1)
            print("Validation IoU score = ", avg_iou)

        epoch_time = time.time() - epoch_st
        remain_time = epoch_time * (num_epochs - 1 - epoch)
        m, s = divmod(remain_time, 60)
        h, m = divmod(m, 60)
        if s != 0:
            train_time = "Remaining training time = %d hours %d minutes %d seconds\n" % (h, m, s)
        else:
            train_time = "Remaining training time : Training completed.\n"
        utils.LOG(train_time)
        scores_list = []
    # print(avg_iou_list)

    fig = plt.figure(figsize=(11, 8))
    ax1 = fig.add_subplot(111)

    ax1.plot(range(num_epochs), avg_scores_per_epoch)
    ax1.set_title("Average validation accuracy vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg. val. accuracy")

    plt.savefig("%s/accuracy_vs_epochs.png" % (ckpt_path))

    plt.clf()

    ax1 = fig.add_subplot(111)

    ax1.plot(range(num_epochs), avg_loss_per_epoch)
    ax1.set_title("Average loss vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Current loss")

    plt.savefig("%s/loss_vs_epochs.png" % (ckpt_path))

    plt.clf()

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rcParams.update({'font.size': 12})

    x = np.arange(1, np.size(avg_iou_list,0)+1)
    ax.plot(x, avg_iou_list, color='blue')
    ax.set_title("Average IoU vs epochs")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU')
    plt.savefig("%s/IoU_vs_epochs.png" % (ckpt_path))
    plt.clf()


    # ax1.plot(range(num_epochs), iou_list)
    # ax1.set_title("Average loss vs epochs")
    # ax1.set_xlabel("Epoch")
    # ax1.set_ylabel("Current loss")
    #
    # plt.savefig("%s/iou_vs_epochs.png" % (ckpt_path))

########################################################################################################################
    gc.collect()