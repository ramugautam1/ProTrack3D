from __future__ import division
import os,time,cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

def preact_conv(inputs, n_filters, kernel_size=[3, 3, 3], dropout_p=0.2):
    """
    Basic pre-activation layer for DenseNets
    Apply successivly BatchNormalization, ReLU nonlinearity, Convolution and
    Dropout (if dropout_p > 0) on the inputs
    """
    #preact = tf.keras.layers.BatchNormalization()(inputs)
    conv = tf.keras.layers.Convolution3D(filters=n_filters, kernel_size=kernel_size, padding='same')(inputs)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation('relu')(conv)
    if dropout_p != 0.0:
      conv = tf.keras.layers.Dropout(rate=dropout_p)(conv)
    return conv

def DenseBlock(stack, n_layers, growth_rate, dropout_p, scope=None):
  """
  DenseBlock for DenseNet and FC-DenseNet
  Arguments:
    stack: input 4D tensor
    n_layers: number of internal layers
    growth_rate: number of feature maps per internal layer
  Returns:
    stack: current stack of feature maps (4D tensor)
    new_features: 4D tensor containing only the new feature maps generated
      in this block
  """
  with tf.name_scope(scope) as sc:
    new_features = []
    for j in range(n_layers):
      # Compute new feature maps
      layer = preact_conv(stack, growth_rate, dropout_p=dropout_p)
      new_features.append(layer)
      # Stack new layer
      stack = tf.concat([stack, layer], axis=-1)
    new_features = tf.concat(new_features, axis=-1)
    return stack, new_features


def TransitionDown(inputs, n_filters, dropout_p=0.2, scope=None):
  """
  Transition Down (TD) for FC-DenseNet
  Apply 1x1 BN + ReLU + conv then 2x2 max pooling
  """
  with tf.name_scope(scope) as sc:
    l = preact_conv(inputs, n_filters, kernel_size=[1, 1, 1], dropout_p=dropout_p)
    l = tf.keras.layers.MaxPool3D(pool_size=[2, 2, 1], strides=[2, 2, 1])(l)
    return l


def TransitionUp(block_to_upsample, skip_connection, n_filters_keep, scope=None):
  """
  Transition Up for FC-DenseNet
  Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection
  """
  with tf.name_scope(scope) as sc:
    # Upsample
    l = tf.keras.layers.UpSampling3D(size=(2, 2, 1))(block_to_upsample)
    l=tf.keras.layers.Convolution3D(filters=n_filters_keep, kernel_size=[3, 3, 3], padding='same')(l)
    l = tf.keras.layers.BatchNormalization()(l)
    l = tf.keras.layers.Activation('relu')(l)
    # Concatenate with skip connection
    l = tf.concat([l, skip_connection], axis=-1)
    return l



def build_fc_densenet(inputs, k_size=[3,3,3], preset_model='FC-DenseNet103', num_classes=2,  n_filters_first_conv=48,  dropout_p=0.2, scope=None):
    """
        Builds the FC-DenseNet model

        Arguments:
          inputs: the input tensor
          preset_model: The model you want to use
          n_classes: number of classes
          n_filters_first_conv: number of filters for the first convolution applied
          n_pool: number of pooling layers = number of transition down = number of transition up
          growth_rate: number of new feature maps created by each layer in a dense block
          n_layers_per_block: number of layers per block. Can be an int or a list of size 2 * n_pool + 1
          dropout_p: dropout rate applied after each convolution (0. for not using)

        Returns:
          Fc-DenseNet model
        """

    if preset_model == 'FC-DenseNet56':
        n_pool = 5
        growth_rate = 12
        n_layers_per_block = 4
    elif preset_model == 'FC-DenseNet67':
        n_pool = 5
        growth_rate = 16
        n_layers_per_block = 5
    elif preset_model == 'FC-DenseNet103':
        n_pool = 5
        growth_rate = 16
        n_layers_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
    else:
        raise ValueError(
            "Unsupported FC-DenseNet model '%s'. This function only supports FC-DenseNet56, FC-DenseNet67, and FC-DenseNet103" % (
                preset_model))

    if type(n_layers_per_block) == list:
        assert (len(n_layers_per_block) == 2 * n_pool + 1)
    elif type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    else:
        raise ValueError

    with tf.variable_scope(scope, preset_model, [inputs]) as sc:
        #####################
        # First Convolution #
        #####################
        # We perform a first convolution.
        stack = tf.keras.layers.Convolution3D(filters=n_filters_first_conv, kernel_size=[3, 3, 3], padding='same')(inputs)

        n_filters = n_filters_first_conv

        #####################
        # Downsampling path #
        #####################

        skip_connection_list = []

        for i in range(n_pool):
            # Dense Block
            stack, _ = DenseBlock(stack, n_layers_per_block[i], growth_rate, dropout_p, scope='denseblock%d' % (i + 1))
            n_filters += growth_rate * n_layers_per_block[i]
            # At the end of the dense block, the current stack is stored in the skip_connections list
            skip_connection_list.append(stack)

            # Transition Down
            stack = TransitionDown(stack, n_filters, dropout_p, scope='transitiondown%d' % (i + 1))

        skip_connection_list = skip_connection_list[::-1]

        #####################
        #     Bottleneck    #
        #####################

        # Dense Block
        # We will only upsample the new feature maps
        stack, block_to_upsample = DenseBlock(stack, n_layers_per_block[n_pool], growth_rate, dropout_p,
                                              scope='denseblock%d' % (n_pool + 1))

        #######################
        #   Upsampling path   #
        #######################

        for i in range(n_pool):
            # Transition Up ( Upsampling + concatenation with the skip connection)
            n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
            stack = TransitionUp(block_to_upsample, skip_connection_list[i], n_filters_keep,
                                 scope='transitionup%d' % (n_pool + i + 1))

            # Dense Block
            # We will only upsample the new feature maps
            stack, block_to_upsample = DenseBlock(stack, n_layers_per_block[n_pool + i + 1], growth_rate, dropout_p,
                                                  scope='denseblock%d' % (n_pool + i + 2))

        #####################
        #      Softmax      #
        #####################
        # net = tf.contrib.layers.conv3d(stack, 2, [1, 1, 1], activation_fn=None, scope='logits')
        net = tf.keras.layers.Convolution3D(filters=2, kernel_size=[1, 1, 1], padding='same')(stack)
        #return net
        return net, stack
