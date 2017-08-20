import os
import time
import shutil
from datetime import timedelta

import numpy as np
import tensorflow as tf

class DenseNet:
    def __init__(self, growth_rate, layers_per_block, keep_prob, is_training=False, reduction=1.0, **kwargs):

        self.growth_rate = growth_rate
        # how many features will be received after first convolution
        # value the same as in the original Torch code
        self.layers_per_block = layers_per_block
        # compression rate at the transition layers
        self.reduction = reduction

        self.keep_prob = keep_prob
        self.is_training = is_training

    def composite_function(self, _input, out_features, kernel_size=3):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        with tf.variable_scope("composite_function"):
            # BN
            output = self.batch_norm(_input)
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = self.conv2d(
                output, out_features=out_features, kernel_size=kernel_size)
            # dropout(in case of training and in case it is no 1.0)
            output = self.dropout(output)
        return output

    def bottleneck(self, _input, out_features):
        with tf.variable_scope("bottleneck"):
            output = self.batch_norm(_input)
            output = tf.nn.relu(output)
            inter_features = out_features * 4
            output = self.conv2d(
                output, out_features=inter_features, kernel_size=1,
                padding='VALID')
            output = self.dropout(output)
        return output

    def add_internal_layer(self, _input, growth_rate):
        """Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """
        # call composite function with 3x3 kernel
        bottleneck_out = self.bottleneck(_input, out_features=growth_rate)
        comp_out = self.composite_function(
            bottleneck_out, out_features=growth_rate, kernel_size=3)
        # concatenate _input with out from composite function

        # output = tf.concat(3, (_input, comp_out))
        output = tf.concat([_input, comp_out], 3)

        return output

    def add_block(self, _input, growth_rate, layers_per_block):
        """Add N H_l internal layers"""
        output = _input
        for layer in range(layers_per_block):
            with tf.variable_scope("layer_%d" % layer):
                output = self.add_internal_layer(output, growth_rate)
        return output

    def transition_layer(self, _input):
        """Call H_l composite function with 1x1 kernel and after average
        pooling
        """
        # call composite function with 1x1 kernel
        out_features = int(int(_input.get_shape()[-1]) * self.reduction)
        output = self.composite_function(
            _input, out_features=out_features, kernel_size=1)
        # run average pooling
        output = self.avg_pool(output, k=2)
        return output

    def upsample_transition_layer(self, _input, ksize, num_output, batch_size):
        shape = _input.get_shape().as_list()
        out_features = int(int(shape[-1]) * self.reduction)
        output = self.composite_function(
            _input, out_features=out_features, kernel_size=1)

        dconv_w = self.weight_variable_xavier([3, 3, num_output, out_features], "dconv_w")
        up_shape = [batch_size, ksize, ksize, num_output]
        output = tf.nn.conv2d_transpose(output, 
            dconv_w, output_shape=up_shape, strides=[1,2,2,1], padding="SAME")

        return output

    def conv2d(self, _input, out_features, kernel_size,
               strides=[1, 1, 1, 1], padding='SAME'):
        in_features = int(_input.get_shape()[-1])
        kernel = self.weight_variable_msra(
            [kernel_size, kernel_size, in_features, out_features],
            name='kernel')
        output = tf.nn.conv2d(_input, kernel, strides, padding)
        return output

    def avg_pool(self, _input, k):
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(_input, ksize, strides, padding)
        return output

    def batch_norm(self, _input):
        output = tf.contrib.layers.batch_norm(
            _input, scale=True, is_training=self.is_training,
            updates_collections=None)
        return output

    def dropout(self, _input):
        if self.keep_prob < 1:
            if self.is_training:
                output = tf.nn.dropout(_input, self.keep_prob)
            else:
                output = _input
        else:
            output = _input
        return output

    def weight_variable_msra(self, shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())

    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)

    def upsample_net(self, net, ksizes, num_outputs_vec, batch_size):
        total_blocks = len(num_outputs_vec)

        for i in range(total_blocks):
            with tf.variable_scope("Block_%d" % i):
                net = self.add_block(net, self.growth_rate, self.layers_per_block)
            # last block exist without transition layer
            with tf.variable_scope("Transition_after_block_%d" % i):
                net = self.upsample_transition_layer(net, ksizes[i], num_outputs_vec[i], batch_size)           
        return net
