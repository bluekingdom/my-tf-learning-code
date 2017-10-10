# -*- coding: utf-8 -*-
# @Author: bluekingdom
# @Date:   2017-08-20 14:42:38
# @Last Modified by:   bluekingdom
# @Last Modified time: 2017-08-23 11:48:27

import tensorflow as tf
import tensorflow.contrib.layers as ly
import numpy as np
import cv2
import os
import random
from functools import partial

from inception_resnet_v1 import inception_resnet_v1, inference
from utils import *
from dense_net import *
from generator_net import get_feature_to_image_net

tf.flags.DEFINE_float("lr", 1e-4, "learning rate (default: 0.001)")
tf.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("checkpoint_file", "", "model restore")
tf.flags.DEFINE_string("pretrain_file", "", "pretrain model")
tf.flags.DEFINE_integer("val_pre_train_batch_iter", 50, "val model")
tf.flags.DEFINE_float("corruption_level", 0.2, "corruption_level")
tf.flags.DEFINE_float("val_ratio", 0.05, "val data ratio")
tf.flags.DEFINE_integer("run_discriminator_per_train_batch_idx", 5, "")
tf.flags.DEFINE_integer("run_generator_per_train_batch_idx", 1, "")
tf.flags.DEFINE_integer("batch_size", 64, "batch size")
tf.flags.DEFINE_integer("total_epoch", 100, "total epoch number")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

z_dim = 100
feature_length =  128
x_h = 160
x_w = 160
channels = 3
batch_size = FLAGS.batch_size
ngf = 256
clamp_lower = -0.01
clamp_upper = 0.01

image_folder = '/home/blue/data/img_align_celeba_10000/'
# image_folder = '/home/mawei/MsCelebV1-Faces-Aligned_160/'
train_files, val_files = load_train_val_data(image_folder, FLAGS.val_ratio)

input_image = tf.placeholder("float", [None, x_h, x_w, channels])
mask = tf.placeholder("float", [None, x_h, x_w, channels], name='mask')
z = tf.placeholder("float", [None, z_dim], name='z')

def get_feature():

    tilde_X = mask * input_image  # corrupted X

    feature, _= inference(tilde_X, 0.8, phase_train=False)

    return feature

def feature_to_image(feature):
    base_output_size = 96
    scale_step = 6
    p_keep_conv = 1.0
    generate_image = get_feature_to_image_net(feature, batch_size, scale_step, base_output_size, channels, p_keep_conv, x_w, x_h)

    return generate_image


# def generator_mlp(z):
#     train = ly.fully_connected(
#         z, ngf, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
#     train = ly.fully_connected(
#         train, ngf, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
#     train = ly.fully_connected(
#         train, ngf, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
#     # train = ly.fully_connected(train, feature_length, activation_fn=tf.nn.tanh, normalizer_fn=ly.batch_norm)
#     train = ly.fully_connected(train, feature_length, 
#         activation_fn=tf.nn.tanh, normalizer_fn=None)
#     return train

def generator_mlp(z):
    train = ly.fully_connected(
        z, ngf, activation_fn=tf.nn.relu)
    train = ly.fully_connected(
        train, ngf, activation_fn=tf.nn.relu)
    train = ly.fully_connected(
        train, ngf, activation_fn=tf.nn.relu)
    train = ly.fully_connected(train, feature_length, activation_fn=tf.nn.tanh)
    return train   

def critic_mlp(x, reuse=False):
    with tf.variable_scope('critic') as scope:
        if reuse:
            scope.reuse_variables()
        net = ly.fully_connected(x, ngf, activation_fn=tf.nn.relu)
        net = ly.fully_connected(net, ngf, activation_fn=tf.nn.relu)
        net = ly.fully_connected(net, ngf, activation_fn=tf.nn.relu)
        logit = ly.fully_connected(net, 1, activation_fn=None)
    return logit

def build_graph():

    X = get_feature()
    generate_image = feature_to_image(X)

    # z_dim = X.get_shape().as_list()[1]

    # noise_dist = tf.contrib.distributions.Normal(-1., 1.)
    # z = noise_dist.sample((batch_size, z_dim))

    with tf.variable_scope('generator'):
        train = generator_mlp(z)
    true_logit = critic_mlp(X)
    fake_logit = critic_mlp(train, reuse=True)
    # c_loss = tf.reduce_mean(fake_logit - true_logit)
    c_loss = tf.reduce_mean(fake_logit) - tf.reduce_mean(true_logit)
    g_loss = -tf.reduce_mean(fake_logit)

    alpha = tf.random_uniform(
        shape=[FLAGS.batch_size, 1], 
        minval=0.,
        maxval=1.
    )
    differences = train - X
    interpolates = X + (alpha*differences)
    gradients = tf.gradients(critic_mlp(interpolates, reuse=True), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    c_loss += 10 * gradient_penalty

    theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

    counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_g = ly.optimize_loss(loss=g_loss, learning_rate=FLAGS.lr,
                    # optimizer=tf.train.RMSPropOptimizer, 
                    optimizer=tf.train.AdamOptimizer, 
                    variables=theta_g, global_step=counter_g)

    counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_c = ly.optimize_loss(loss=c_loss, learning_rate=FLAGS.lr,
                    # optimizer=tf.train.RMSPropOptimizer, 
                    optimizer=tf.train.AdamOptimizer, 
                    variables=theta_c, global_step=counter_c)

    clipped_var_c = [tf.assign(var, tf.clip_by_value(var, clamp_lower, clamp_upper)) for var in theta_c]

    # merge the clip operations on critic variables
    with tf.control_dependencies([opt_c]):
        opt_c = tf.tuple(clipped_var_c)

    return opt_g, opt_c, c_loss, g_loss, generate_image


def train():
    opt_g, opt_c, c_loss, g_loss, generate_image = build_graph()

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    with tf.Session() as sess:

        # you need to initialize all variables
        tf.global_variables_initializer().run()

        pretrain_vars = [v for v in tf.trainable_variables() if v.name.startswith('InceptionResnetV1')]
        pretrain_vars += [v for v in tf.trainable_variables() if v.name.startswith('generate_net')]
        for op in pretrain_vars:
            print(op.name)
        restore_network(saver, sess, FLAGS.checkpoint_file, FLAGS.num_checkpoints, FLAGS.pretrain_file, pretrain_vars)

        train_imgs_batch = batch_iter(train_files, batch_size, num_epochs=1)

        train_batch_idx = 0
        min_loss = 1e20
        for imgs_batch in train_imgs_batch:
            noise = np.random.normal(size=(imgs_batch.shape[0], z_dim))
            mask_np = np.random.binomial(1, 1 - FLAGS.corruption_level, imgs_batch.shape)
            train_batch_idx += 1

            if train_batch_idx % FLAGS.val_pre_train_batch_iter == FLAGS.val_pre_train_batch_iter - 1:
                val_batch = batch_iter(val_files, batch_size, num_epochs=1, shuffle=True)
                val_loss = 0
                val_iter_count = 0

                for val_b in val_batch:
                    if val_iter_count > 10: break
                    val_iter_count += 1
                    val_mask_np = np.random.binomial(1, 1 - FLAGS.corruption_level, val_b.shape)

                    val_loss += sess.run(tf.reduce_mean(c_loss), feed_dict={input_image: val_b, mask: val_mask_np, z: noise})

                val_loss /= val_iter_count
                print('iter: %6d, val_loss: %0.5f, min_loss: %0.5f' % (train_batch_idx, val_loss, min_loss))
                if abs(min_loss) > abs(val_loss):
                    min_loss = val_loss
                    path = saver.save(sess, "models/feature_model", global_step=train_batch_idx)
                    print("Saved model checkpoint to {}".format(path))
                    pass


                predicted_imgs = sess.run(generate_image, feed_dict={input_image: imgs_batch, mask: mask_np, z: noise})
                vis(predicted_imgs, 'feature_pred_%d' % train_batch_idx)
                vis(np.array(imgs_batch, dtype=np.float32), 'feature_in_%d' % train_batch_idx)
                pass

            if train_batch_idx > 0:
                gl_m = 0
                for i in range(FLAGS.run_generator_per_train_batch_idx):
                    _, gl = sess.run([opt_g, g_loss], feed_dict={z: noise})
                    gl_m += gl
                gl_m /= FLAGS.run_generator_per_train_batch_idx
                pass

            cl_m = 0
            # c_count = FLAGS.run_discriminator_per_train_batch_idx if train_batch_idx > 25 else 20
            c_count = FLAGS.run_discriminator_per_train_batch_idx
            for i in range(c_count):
                _, cl = sess.run([opt_c, c_loss], feed_dict={input_image: imgs_batch, mask: mask_np, z: noise})
                cl_m += cl
            cl_m /= c_count



            print('iter: %5d, g_loss: %0.5f, c_loss: %0.5f' % (train_batch_idx, gl_m, cl_m))

            pass
        pass




if __name__ == '__main__':
    train()
