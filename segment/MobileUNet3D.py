# import os,time,cv2
# import tensorflow as tf
# import tensorflow.contrib.slim as slim
# import numpy as np
#
# def ConvBlock(inputs, n_filters, kernel_size=[3, 3, 3]):
# 	"""
# 	Builds the conv block for MobileNets
# 	Apply successivly a 2D convolution, BatchNormalization relu
# 	"""
# 	# Skip pointwise by setting num_outputs=Non
# 	net = tf.keras.layers.Convolution3D(filters=n_filters, kernel_size=[1, 1, 1], padding='same')(inputs)
# 	net = tf.keras.layers.BatchNormalization(fused=True)(net)
# 	net = tf.keras.layers.Activation('relu')(net)
# 	return net
#
# def DepthwiseSeparableConvBlock(inputs, n_filters, kernel_size=[3, 3, 3]):
# 	"""
# 	Builds the Depthwise Separable conv block for MobileNets
# 	Apply successivly a 2D separable convolution, BatchNormalization relu, conv, BatchNormalization, relu
# 	"""
# 	# Skip pointwise by setting num_outputs=None
# 	net = tf.keras.layers.Convolution3D(filters=n_filters, kernel_size=kernel_size, padding='same')(inputs)
#
# 	net = tf.keras.layers.BatchNormalization(fused=True)(net)
# 	net = tf.keras.layers.Activation('relu')(net)
# 	net = tf.keras.layers.Convolution3D(filters=n_filters, kernel_size=[1, 1, 1], padding='same')(net)
# 	net = tf.keras.layers.BatchNormalization(fused=True)(net)
# 	net = tf.keras.layers.Activation('relu')(net)
# 	return net
#
# def conv_transpose_block(inputs, n_filters, kernel_size=[3, 3, 3]):
# 	"""
# 	Basic conv transpose block for Encoder-Decoder upsampling
# 	Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
# 	"""
# 	net = tf.keras.layers.UpSampling3D(size=(2, 2, 1))(inputs)
# 	net = tf.keras.layers.Convolution3D(filters=n_filters, kernel_size=[3, 3, 3], padding='same')(net)
# 	net = tf.keras.layers.BatchNormalization()(net)
# 	net = tf.keras.layers.Activation('relu')(net)
# 	return net
#
# def build_mobile_unet3D(inputs, preset_model, num_classes):
#
# 	has_skip = False
# 	if preset_model == "MobileUNet3D":
# 		has_skip = False
# 	elif preset_model == "MobileUNet3D-Skip":
# 		has_skip = True
# 	else:
# 		raise ValueError("Unsupported MobileUNet model '%s'. This function only supports MobileUNet and MobileUNet-Skip" % (preset_model))
#
#     #####################
# 	# Downsampling path #
# 	#####################
# 	net = ConvBlock(inputs, 64)
# 	net = DepthwiseSeparableConvBlock(net, 64)
# 	net = tf.keras.layers.MaxPool3D(pool_size=[2, 2, 1], strides=[2, 2, 1])(net)
# 	skip_1 = net
#
# 	net = DepthwiseSeparableConvBlock(net, 128)
# 	net = DepthwiseSeparableConvBlock(net, 128)
# 	net = tf.keras.layers.MaxPool3D(pool_size=[2, 2, 1], strides=[2, 2, 1])(net)
# 	skip_2 = net
#
# 	net = DepthwiseSeparableConvBlock(net, 256)
# 	net = DepthwiseSeparableConvBlock(net, 256)
# 	net = DepthwiseSeparableConvBlock(net, 256)
# 	net = tf.keras.layers.MaxPool3D(pool_size=[2, 2, 1], strides=[2, 2, 1])(net)
# 	skip_3 = net
#
# 	net = DepthwiseSeparableConvBlock(net, 512)
# 	net = DepthwiseSeparableConvBlock(net, 512)
# 	net = DepthwiseSeparableConvBlock(net, 512)
# 	net = tf.keras.layers.MaxPool3D(pool_size=[2, 2, 1], strides=[2, 2, 1])(net)
# 	skip_4 = net
#
# 	net = DepthwiseSeparableConvBlock(net, 512)
# 	net = DepthwiseSeparableConvBlock(net, 512)
# 	net = DepthwiseSeparableConvBlock(net, 512)
# 	net = tf.keras.layers.MaxPool3D(pool_size=[2, 2, 1], strides=[2, 2, 1])(net)
#
#
# 	#####################
# 	# Upsampling path #
# 	#####################
# 	net = conv_transpose_block(net, 512)
# 	net = DepthwiseSeparableConvBlock(net, 512)
# 	net = DepthwiseSeparableConvBlock(net, 512)
# 	net = DepthwiseSeparableConvBlock(net, 512)
# 	if has_skip:
# 		net = tf.add(net, skip_4)
#
# 	net = conv_transpose_block(net, 512)
# 	net = DepthwiseSeparableConvBlock(net, 512)
# 	net = DepthwiseSeparableConvBlock(net, 512)
# 	net = DepthwiseSeparableConvBlock(net, 256)
# 	if has_skip:
# 		net = tf.add(net, skip_3)
#
# 	net = conv_transpose_block(net, 256)
# 	net = DepthwiseSeparableConvBlock(net, 256)
# 	net = DepthwiseSeparableConvBlock(net, 256)
# 	net = DepthwiseSeparableConvBlock(net, 128)
# 	if has_skip:
# 		net = tf.add(net, skip_2)
#
# 	net = conv_transpose_block(net, 128)
# 	net = DepthwiseSeparableConvBlock(net, 128)
# 	net = DepthwiseSeparableConvBlock(net, 64)
# 	if has_skip:
# 		net = tf.add(net, skip_1)
#
# 	net = conv_transpose_block(net, 64)
# 	net = DepthwiseSeparableConvBlock(net, 64)
# 	net = DepthwiseSeparableConvBlock(net, 64)
#
# 	#####################
# 	#      Softmax      #
# 	#####################
# 	net = tf.contrib.layers.conv3d(net, num_classes, [1, 1, 1], activation_fn=None, scope='logits')
# 	return net



import os
import time
import cv2
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import numpy as np

def ConvBlock(inputs, n_filters, kernel_size=[3, 3, 3]):
    """
    Builds the conv block for MobileNets
    Apply successively a 2D convolution, BatchNormalization relu
    """
    # Skip pointwise by setting num_outputs=None
    net = tf.keras.layers.Conv3D(filters=n_filters, kernel_size=[1, 1, 1], padding='same')(inputs)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation('relu')(net)
    return net

def DepthwiseSeparableConvBlock(inputs, n_filters, kernel_size=[3, 3, 3]):
    """
    Builds the Depthwise Separable conv block for MobileNets
    Apply successively a 2D separable convolution, BatchNormalization relu, conv, BatchNormalization, relu
    """
    # Skip pointwise by setting num_outputs=None
    net = tf.keras.layers.Conv3D(filters=n_filters, kernel_size=kernel_size, padding='same')(inputs)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation('relu')(net)
    net = tf.keras.layers.Conv3D(filters=n_filters, kernel_size=[1, 1, 1], padding='same')(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation('relu')(net)
    return net

def conv_transpose_block(inputs, n_filters, kernel_size=[3, 3, 3]):
    """
    Basic conv transpose block for Encoder-Decoder upsampling
    Apply successively Transposed Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = tf.keras.layers.UpSampling3D(size=(2, 2, 1))(inputs)
    net = tf.keras.layers.Conv3D(filters=n_filters, kernel_size=[3, 3, 3], padding='same')(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation('relu')(net)
    return net

def build_mobile_unet3D(inputs, preset_model, num_classes):

    has_skip = False
    if preset_model == "MobileUNet3D":
        has_skip = True
    elif preset_model == "MobileUNet3D-Skip":
        has_skip = True
    else:
        raise ValueError("Unsupported MobileUNet model '%s'. This function only supports MobileUNet and MobileUNet-Skip" % (preset_model))

    #####################
    # Downsampling path #
    #####################
    net = ConvBlock(inputs, 64)
    net = DepthwiseSeparableConvBlock(net, 64)
    net = tf.keras.layers.MaxPool3D(pool_size=[2, 2, 1], strides=[2, 2, 1])(net)
    skip_1 = net

    net = DepthwiseSeparableConvBlock(net, 128)
    net = DepthwiseSeparableConvBlock(net, 128)
    net = tf.keras.layers.MaxPool3D(pool_size=[2, 2, 1], strides=[2, 2, 1])(net)
    skip_2 = net

    net = DepthwiseSeparableConvBlock(net, 256)
    net = DepthwiseSeparableConvBlock(net, 256)
    net = DepthwiseSeparableConvBlock(net, 256)
    net = tf.keras.layers.MaxPool3D(pool_size=[2, 2, 1], strides=[2, 2, 1])(net)
    skip_3 = net

    net = DepthwiseSeparableConvBlock(net, 512)
    net = DepthwiseSeparableConvBlock(net, 512)
    net = DepthwiseSeparableConvBlock(net, 512)
    net = tf.keras.layers.MaxPool3D(pool_size=[2, 2, 1], strides=[2, 2, 1])(net)
    skip_4 = net

    net = DepthwiseSeparableConvBlock(net, 512)
    net = DepthwiseSeparableConvBlock(net, 512)
    net = DepthwiseSeparableConvBlock(net, 512)
    net = tf.keras.layers.MaxPool3D(pool_size=[2, 2, 1], strides=[2, 2, 1])(net)

    #####################
    # Upsampling path #
    #####################
    net = conv_transpose_block(net, 512)
    net = DepthwiseSeparableConvBlock(net, 512)
    net = DepthwiseSeparableConvBlock(net, 512)
    net = DepthwiseSeparableConvBlock(net, 512)
    if has_skip:
        net = tf.add(net, skip_4)

    net = conv_transpose_block(net, 512)
    net = DepthwiseSeparableConvBlock(net, 512)
    net = DepthwiseSeparableConvBlock(net, 512)
    net = DepthwiseSeparableConvBlock(net, 256)
    if has_skip:
        net = tf.add(net, skip_3)

    net = conv_transpose_block(net, 256)
    net = DepthwiseSeparableConvBlock(net, 256)
    net = DepthwiseSeparableConvBlock(net, 256)
    net = DepthwiseSeparableConvBlock(net, 128)
    if has_skip:
        net = tf.add(net, skip_2)

    net = conv_transpose_block(net, 128)
    net = DepthwiseSeparableConvBlock(net, 128)
    net = DepthwiseSeparableConvBlock(net, 64)
    if has_skip:
        net = tf.add(net, skip_1)

    net = conv_transpose_block(net, 64)
    net = DepthwiseSeparableConvBlock(net, 64)
    net = DepthwiseSeparableConvBlock(net, 64)

    #####################
    #      Softmax      #
    #####################
    net = tf.keras.layers.Conv3D(num_classes, [1, 1, 1], activation=None, padding='same')(net) #,name='logits'
    return net,net
