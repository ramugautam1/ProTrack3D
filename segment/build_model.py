import tensorflow as tf
# import keras
# from keras.models import Model
# from keras.layers.merge import concatenate
# from keras.layers import Input, Convolution3D, MaxPooling3D, UpSampling3D
# from keras.layers import Reshape, Activation, Permute
# # from keras.layers.normalization import BatchNormalization
# from keras.layers.normalization import BatchNormalization

def build_mymodel(inp_shape, k_size=[3,3,3]):
    merge_axis = -1 # Feature maps are concatenated along last axis (for tf backend)
    #data = tf.keras.layers.Input(shape=inp_shape.shape, input_tensor=inp_shape)
    conv1 = tf.keras.layers.Convolution3D(padding='same', filters=64, kernel_size=k_size)(inp_shape)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Activation('relu')(conv1)
    conv2 = tf.keras.layers.Convolution3D(padding='same', filters=64, kernel_size=k_size)(conv1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.Activation('relu')(conv2)
    pool1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(conv2)

    conv3 = tf.keras.layers.Convolution3D(padding='same', filters=128, kernel_size=k_size)(pool1)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.Activation('relu')(conv3)
    conv4 = tf.keras.layers.Convolution3D(padding='same', filters=128, kernel_size=k_size)(conv3)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    conv4 = tf.keras.layers.Activation('relu')(conv4)
    pool2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(conv4)

    conv5 = tf.keras.layers.Convolution3D(padding='same', filters=128, kernel_size=k_size)(pool2)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    conv5 = tf.keras.layers.Activation('relu')(conv5)
    conv6 = tf.keras.layers.Convolution3D(padding='same', filters=128, kernel_size=k_size)(conv5)
    conv6 = tf.keras.layers.BatchNormalization()(conv6)
    conv6 = tf.keras.layers.Activation('relu')(conv6)
    pool3 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(conv6)

    conv7 = tf.keras.layers.Convolution3D(padding='same', filters=256, kernel_size=k_size)(pool3)
    conv7 = tf.keras.layers.BatchNormalization()(conv7)
    conv7 = tf.keras.layers.Activation('relu')(conv7)
    conv8 = tf.keras.layers.Convolution3D(padding='same', filters=256, kernel_size=k_size)(conv7)
    conv8 = tf.keras.layers.BatchNormalization()(conv8)
    conv8 = tf.keras.layers.Activation('relu')(conv8)
    pool4 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(conv8)

    conv9 = tf.keras.layers.Convolution3D(padding='same', filters=256, kernel_size=k_size)(pool4)
    conv9 = tf.keras.layers.BatchNormalization()(conv9)
    conv9 = tf.keras.layers.Activation('relu')(conv9)

    up1 = tf.keras.layers.UpSampling3D(size=(2, 2, 1))(conv9)
    conv10 = tf.keras.layers.Convolution3D(padding='same', filters=256, kernel_size=k_size)(up1)
    conv10 = tf.keras.layers.BatchNormalization()(conv10)
    conv10 = tf.keras.layers.Activation('relu')(conv10)
    conv11 = tf.keras.layers.Convolution3D(padding='same', filters=256, kernel_size=k_size)(conv10)
    conv11 = tf.keras.layers.BatchNormalization()(conv11)
    conv11 = tf.keras.layers.Activation('relu')(conv11)
    merged1 = tf.keras.layers.concatenate([conv11, conv8], axis=merge_axis)
    conv12 = tf.keras.layers.Convolution3D(padding='same', filters=256, kernel_size=k_size)(merged1)
    conv12 = tf.keras.layers.BatchNormalization()(conv12)
    conv12 = tf.keras.layers.Activation('relu')(conv12)

    up2 = tf.keras.layers.UpSampling3D(size=(2, 2, 1))(conv12)
    conv13 = tf.keras.layers.Convolution3D(padding='same', filters=128, kernel_size=k_size)(up2)
    conv13 = tf.keras.layers.BatchNormalization()(conv13)
    conv13 = tf.keras.layers.Activation('relu')(conv13)
    conv14 = tf.keras.layers.Convolution3D(padding='same', filters=128, kernel_size=k_size)(conv13)
    conv14 = tf.keras.layers.BatchNormalization()(conv14)
    conv14 = tf.keras.layers.Activation('relu')(conv14)
    merged2 = tf.keras.layers.concatenate([conv14, conv6], axis=merge_axis)
    conv15 = tf.keras.layers.Convolution3D(padding='same', filters=128, kernel_size=k_size)(merged2)
    conv15 = tf.keras.layers.BatchNormalization()(conv15)
    conv15 = tf.keras.layers.Activation('relu')(conv15)

    up3 = tf.keras.layers.UpSampling3D(size=(2, 2, 1))(conv15)
    conv16 = tf.keras.layers.Convolution3D(padding='same', filters=128, kernel_size=k_size)(up3)
    conv16 = tf.keras.layers.BatchNormalization()(conv16)
    conv16 = tf.keras.layers.Activation('relu')(conv16)
    conv17 = tf.keras.layers.Convolution3D(padding='same', filters=128, kernel_size=k_size)(conv16)
    conv17 = tf.keras.layers.BatchNormalization()(conv17)
    conv17 = tf.keras.layers.Activation('relu')(conv17)
    merged3 = tf.keras.layers.concatenate([conv17, conv4], axis=merge_axis)
    conv18 = tf.keras.layers.Convolution3D(padding='same', filters=128, kernel_size=k_size)(merged3)
    conv18 = tf.keras.layers.BatchNormalization()(conv18)
    conv18 = tf.keras.layers.Activation('relu')(conv18)

    up4 = tf.keras.layers.UpSampling3D(size=(2, 2, 1))(conv18)
    conv19 = tf.keras.layers.Convolution3D(padding='same', filters=128, kernel_size=k_size)(up4)
    conv19 = tf.keras.layers.BatchNormalization()(conv19)
    conv19 = tf.keras.layers.Activation('relu')(conv19)
    conv20 = tf.keras.layers.Convolution3D(padding='same', filters=128, kernel_size=k_size)(conv19)
    conv20 = tf.keras.layers.BatchNormalization()(conv20)
    conv20 = tf.keras.layers.Activation('relu')(conv20)
    merged4 = tf.keras.layers.concatenate([conv20, conv2], axis=merge_axis)
    conv21 = tf.keras.layers.Convolution3D(padding='same', filters=128, kernel_size=k_size)(merged4)
    conv21 = tf.keras.layers.BatchNormalization()(conv21)
    conv21 = tf.keras.layers.Activation('relu')(conv21)

    #conv22 =tf.keras.layers. Convolution3D(padding='same', filters=2, kernel_size=[1, 1, 1])(conv21)

    #output = tf.keras.layers.Reshape([-1, 2])(conv22)
    #output = tf.keras.layers.Activation('softmax')(output)
    #output = tf.keras.layers.Reshape(inp_shape[:-1] + (2,))(output)
    net = tf.contrib.layers.conv3d(conv21, 2, [1, 1, 1], activation_fn=None, scope='logits')
    return net





