import os
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras 
import tensorflow as tf
import numpy.matlib
from PIL import Image
from keras import backend as K
from scipy.special import softmax
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import bottleneck
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Sequential



# class Bottleneck(keras.Model):
#   def __init__(
#       self,
#       expansion,
#       stride,
#       block_id,
#       filters,
#       alpha=1,
#       ):
#     super(Bottleneck,self).__init__(name = "Bottleneck_" + block_id)
#     self.stride = stride
#     self.expansion = expansion
#     self.alpha = alpha
#     self.output_channels = self.alpha * filters
#     self.out = None # there was some problem with the eager execution

#     prefix =  'Bottleneck_{}_'.format(block_id)
#     self.prefix = prefix
#     # expansion
#     self.expand_BN = layers.BatchNormalization(name = prefix + 'expand_BN')
#     self.expand_ReLU = layers.ReLU(max_value=6, name = prefix + 'expand_ReLU')

#     #conv
#     self.Conv = layers.DepthwiseConv2D(
#         kernel_size = 3,
#         padding='same',
#         strides = self.stride,
#         use_bias = False,
#         name = prefix + 'conv')
#     self.Conv_BN = layers.BatchNormalization(name = prefix + 'conv_BN')
#     self.Conv_ReLU = layers.ReLU(max_value=6, name = prefix + 'conv_ReLU')

#     #project
#     self.project = layers.Conv2D(
#         filters = self.output_channels,
#         kernel_size = 1,
#         use_bias = False,
#         name = 'contract')
#     self.project_BN = layers.BatchNormalization(name = prefix + 'contract_BN')

#     # dimensions need to be the same for residual connection
#     self.residual = layers.Add(name=prefix + 'residual')
  
#   def build(self, input_shape):
#     self.d = input_shape[-1]
    
#     self.expand = layers.Conv2D(
#         filters = self.expansion*self.d,
#         kernel_size = 1,
#         use_bias = False,
#         name = self.prefix+'expand')

      
#   def call(self, inputs):

#     x = self.expand(inputs)
#     x = self.expand_BN(x)
#     x = self.expand_ReLU(x)
#     self.out = x
    
#     x = self.Conv(x)
#     x = self.Conv_BN(x)
#     x = self.Conv_ReLU(x)

#     x = self.project(x)
#     x = self.project_BN(x)

#     if self.output_channels == self.d and self.stride == 1:
#       x = self.residual([inputs,x])

#     return x

#   def model(self):
#       x = keras.Input(shape=(28,28,3))
#       return keras.Model(inputs=[x], outputs=self.call(x))
    
    
    
    
    
# #using the architecture mentioned in the paper
# class MobileNetv2(keras.Model):
#   def __init__(self, k = 11):
#     super(MobileNetv2,self).__init__()
#     self.conv_inp = layers.Conv2D(
#         filters = 32,
#         kernel_size = 3,
#         strides = (2,2),
#         padding='valid',
#         use_bias = False,
#         name = 'conv'
#     )
#     self.k = k    

#     self.pad = layers.ZeroPadding2D(padding=2,name='pad')
#     self.BN = layers.BatchNormalization(name='BN')
#     self.ReLU = layers.ReLU(max_value = 6, name = 'ReLU')
    
#     self.B1_1 = Bottleneck(expansion = 1, filters = 16, stride = 1, block_id = 'B1_1')

#     self.B2_1 = Bottleneck(expansion = 6, filters = 24, stride = 2, block_id = 'B2_1')
#     self.B2_2 = Bottleneck(expansion = 6, filters = 24, stride = 1, block_id = 'B2_2')

#     self.B3_1 = Bottleneck(expansion = 6, filters = 32, stride = 2, block_id = 'B3_1')
#     self.B3_2 = Bottleneck(expansion = 6, filters = 32, stride = 1, block_id = 'B3_2')
#     self.B3_3 = Bottleneck(expansion = 6, filters = 32, stride = 1, block_id = 'B3_3')

#     self.B4_1 = Bottleneck(expansion = 6, filters = 64, stride = 2, block_id = 'B4_1')
#     self.B4_2 = Bottleneck(expansion = 6, filters = 64, stride = 1, block_id = 'B4_2')
#     self.B4_3 = Bottleneck(expansion = 6, filters = 64, stride = 1, block_id = 'B4_3')
#     self.B4_4 = Bottleneck(expansion = 6, filters = 64, stride = 1, block_id = 'B4_4')

#     self.B5_1 = Bottleneck(expansion = 6, filters = 96, stride = 1, block_id = 'B5_1')
#     self.B5_2 = Bottleneck(expansion = 6, filters = 96, stride = 1, block_id = 'B5_2')
#     self.B5_3 = Bottleneck(expansion = 6, filters = 96, stride = 1, block_id = 'B5_3')

#     self.B6_1 = Bottleneck(expansion = 6, filters = 160, stride = 2, block_id = 'B6_1')
#     self.B6_2 = Bottleneck(expansion = 6, filters = 160, stride = 1, block_id = 'B6_2')
#     self.B6_3 = Bottleneck(expansion = 6, filters = 160, stride = 1, block_id = 'B6_3')

#     self.B7_1 = Bottleneck(expansion = 6, filters = 320, stride = 1, block_id = 'B7_1')

#     self.conv_out = layers.Conv2D(
#         filters = 1280,
#         kernel_size = 1,
#         strides = (1,1),
#         use_bias = False,
#         name = 'conv_out'
#     )
#     self.avgpool = layers.AveragePooling2D(
#         pool_size = (7,7),
#         name='avg_pool'
#         )
    
#     self.conv_seg = layers.Conv2D(
#         filters = self.k,
#         kernel_size = 1,
#         strides = (1,1),
#         use_bias = False,
#         name = 'conv_seg'
#     )

#   def call(self, inputs):
#     x = self.conv_inp(inputs)
#     x = self.BN(x)
#     x = self.ReLU(x)

#     x = self.B1_1(x)
#     x = self.B2_1(x)
#     x = self.B2_2(x)

#     x = self.B3_1(x)
#     x = self.B3_2(x)
#     x = self.B3_3(x)
    
#     x = self.B4_1(x)
#     x = self.B4_2(x)
#     x = self.B4_3(x)
#     x = self.B4_4(x)
    
#     x = self.B5_1(x)
#     x = self.B5_2(x)
#     x = self.B5_3(x)
    
#     x = self.B6_1(x)
#     x = self.B6_2(x)
#     x = self.B6_3(x)
    
#     x = self.B7_1(x)

#     x = self.conv_out(x)
#     x = self.avgpool(x)
#     c4 = self.conv_seg(x)

#     return c4

#   def model(self):
#       x = keras.Input(shape=(224,224,3))

#       return keras.Model(inputs=x, outputs=self.call(x))



def create_vgg16_layers():
    vgg16_conv4 = [
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPool2D(2, 2, padding='same'),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPool2D(2, 2, padding='same'),

        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.MaxPool2D(2, 2, padding='same'),

        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.MaxPool2D(2, 2, padding='same'),

        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
    ]

    x = layers.Input(shape=[None, None, 3])
    out = x
    for layer in vgg16_conv4:
        out = layer(out)

    vgg16_conv4 = tf.keras.Model(x, out)

    vgg16_conv7 = [
        # Difference from original VGG16:
        # 5th maxpool layer has kernel size = 3 and stride = 1
        layers.MaxPool2D(3, 1, padding='same'),
        # atrous conv2d for 6th block
        layers.Conv2D(1024, 3, padding='same',
                      dilation_rate=6, activation='relu'),
        layers.Conv2D(1024, 1, padding='same', activation='relu'),
    ]

    x = layers.Input(shape=[None, None, 512])
    out = x
    for layer in vgg16_conv7:
        out = layer(out)

    vgg16_conv7 = tf.keras.Model(x, out)

    return vgg16_conv4, vgg16_conv7


def create_extra_layers():
    """ Create extra layers
        8th to 11th blocks
    """
    extra_layers = [
        # 8th block output shape: B, 512, 10, 10
        Sequential([
            layers.Conv2D(256, 1, activation='relu'),
            layers.Conv2D(512, 3, strides=2, padding='same',
                          activation='relu'),
        ]),
        # 9th block output shape: B, 256, 5, 5
        Sequential([
            layers.Conv2D(128, 1, activation='relu'),
            layers.Conv2D(256, 3, strides=2, padding='same',
                          activation='relu'),
        ]),
        # 10th block output shape: B, 256, 3, 3
        Sequential([
            layers.Conv2D(128, 1, activation='relu'),
            layers.Conv2D(256, 3, activation='relu'),
        ]),
        # 11th block output shape: B, 256, 1, 1
        Sequential([
            layers.Conv2D(128, 1, activation='relu'),
            layers.Conv2D(256, 3, activation='relu'),
        ]),
        # 12th block output shape: B, 256, 1, 1
        Sequential([
            layers.Conv2D(128, 1, activation='relu'),
            layers.Conv2D(256, 4, activation='relu'),
        ])
    ]

    return extra_layers


def create_conf_head_layers(num_classes):
    """ Create layers for classification
    """
    conf_head_layers = [
        layers.Conv2D(4 * num_classes, kernel_size=3,
                      padding='same'),  # for 4th block
        layers.Conv2D(6 * num_classes, kernel_size=3,
                      padding='same'),  # for 7th block
        layers.Conv2D(6 * num_classes, kernel_size=3,
                      padding='same'),  # for 8th block
        layers.Conv2D(6 * num_classes, kernel_size=3,
                      padding='same'),  # for 9th block
        layers.Conv2D(4 * num_classes, kernel_size=3,
                      padding='same'),  # for 10th block
        layers.Conv2D(4 * num_classes, kernel_size=3,
                      padding='same'),  # for 11th block
        layers.Conv2D(4 * num_classes, kernel_size=1)  # for 12th block
    ]

    return conf_head_layers


def create_loc_head_layers():
    """ Create layers for regression
    """
    loc_head_layers = [
        layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(4 * 4, kernel_size=1)
    ]

    return loc_head_layers

