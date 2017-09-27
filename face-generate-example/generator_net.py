# -*- coding: utf-8 -*-
"""
* @File Name:   		generator_net.py
* @Author:				Wang Yang
* @Created Date:		2017-09-27 20:38:32
* @Last Modified Data:	2017-09-27 20:49:45
* @Desc:					
*
"""
import tensorflow as tf
import numpy as np

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def get_generate_net(net, batch_size, scale_step, base_output_size, channels, x_w, x_h):

    weights = {}
    print('init weights\n')
    for i in range(1, scale_step + 1, 1):
        output_scale = 2 ** (i-2)
        o1 = output_scale * base_output_size if i > 1 else channels
        o2 = 2 * output_scale * base_output_size
        shape = [3, 3, int(o1), int(o2)]
        # print(i, shape)
        weights['rw%d' % i] = init_weights(shape)
        shape = [3, 3, int(o2), int(o2)]
        weights['rw%d_conv' % i] = init_weights(shape)
        weights['rw%d_bias' % i] = init_weights([int(o2)])
    weights['rw6'] = init_weights([3, 3, 1536, 1792])
    weights['rw6_conv'] = init_weights([3, 3, 1792, 1792])
    weights['rw6_bias'] = init_weights([1792])

    # print('weights:')
    # for (k,v) in weights.items():
    #     print(k, v)

    for i in range(scale_step - 1, -1, -1):
        #  160  80  40  20  10    5
        #    c  96 192 384 768 1536
        size = float(2 ** i)
        output_size = base_output_size * size / 2 if i > 0 else channels
        up_shape = [batch_size, int(np.ceil(x_h / size)), int(np.ceil(x_w / size)), int(output_size)]
        print(i, up_shape)

        net = tf.nn.conv2d(net, weights['rw%d_conv' % (i+1)], strides=[1,1,1,1], padding="SAME")
        net = tf.nn.bias_add(net, weights['rw%d_bias' % (i+1)])
        net = tf.nn.relu(net)

        net = tf.nn.conv2d_transpose(net, weights['rw%d' % (i+1)], output_shape=up_shape, strides=[1,2,2,1], padding="SAME")

        if i != 0:
            net = tf.nn.relu(net)
        else:
            net = tf.nn.sigmoid(net)
        pass

    return net

def get_feature_to_image_net(feature, batch_size, scale_step, base_output_size, channels, p_keep_conv, x_w, x_h):

    with tf.name_scope("generate_net") as sc:
        f_net_shape = feature.get_shape().as_list()
        print(f_net_shape)

        fc_shape = f_net_shape[1]
        feature1 = tf.contrib.layers.fully_connected(feature, fc_shape, scope=sc + 'feature1')
        feature1 = tf.nn.dropout(feature1, p_keep_conv)
        feature2 = tf.contrib.layers.fully_connected(feature1, 3 * 3 * 1792, scope=sc + 'feature2')
        f_net_shape = [batch_size, 3, 3, 1792]
        up_sacle_net = tf.reshape(feature2, f_net_shape)

        # use normal upsample net
        net = get_generate_net(up_sacle_net, batch_size, scale_step, base_output_size, channels, x_w, x_h)
    return net
