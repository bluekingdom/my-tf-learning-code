# -*- coding: utf-8 -*-
# @Author: bluekingdom
# @Date:   2017-08-20 14:42:38
# @Last Modified by:   bluekingdom
# @Last Modified time: 2017-08-22 17:15:50

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

tf.flags.DEFINE_float("lr", 0.001, "learning rate (default: 0.001)")
tf.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("checkpoint_file", "", "model restore")
tf.flags.DEFINE_integer("val_pre_train_batch_iter", 50, "val model")
tf.flags.DEFINE_float("corruption_level", 0.2, "corruption_level")
tf.flags.DEFINE_float("val_ratio", 0.05, "val data ratio")
tf.flags.DEFINE_integer("run_discriminator_per_train_batch_idx", 1, "")
tf.flags.DEFINE_integer("run_generator_per_train_batch_idx", 100, "")
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

def get_feature():

    tilde_X = mask * input_image  # corrupted X

    feature, _= inference(tilde_X, 0.8, phase_train=False)

    return feature

def generator_mlp(z):
    train = ly.fully_connected(
        z, ngf, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
    train = ly.fully_connected(
        train, ngf, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
    train = ly.fully_connected(
        train, ngf, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
    train = ly.fully_connected(
        train, feature_length, activation_fn=tf.nn.tanh, normalizer_fn=ly.batch_norm)
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

    # X = tf.placeholder("float", [None, feature_length], name='X')
    X = get_feature()
    z_dim = X.get_shape().as_list()[1]

    noise_dist = tf.contrib.distributions.Normal(0., 1.)
    z = noise_dist.sample((batch_size, z_dim))

    with tf.variable_scope('generator'):
        train = generator_mlp(z)
    true_logit = critic_mlp(X)
    fake_logit = critic_mlp(train, reuse=True)
    # c_loss = tf.reduce_mean(fake_logit - true_logit)
    # g_loss = tf.reduce_mean(-fake_logit)
    c_loss = tf.reduce_mean(-fake_logit + true_logit)
    g_loss = tf.reduce_mean(fake_logit)

    theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

    counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_g = ly.optimize_loss(loss=g_loss, learning_rate=FLAGS.lr,
                    optimizer=tf.train.RMSPropOptimizer, 
                    variables=theta_g, global_step=counter_g)

    counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_c = ly.optimize_loss(loss=c_loss, learning_rate=FLAGS.lr,
                    optimizer=tf.train.RMSPropOptimizer, 
                    variables=theta_c, global_step=counter_c)

    clipped_var_c = [tf.assign(var, tf.clip_by_value(var, clamp_lower, clamp_upper)) for var in theta_c]

    # merge the clip operations on critic variables
    with tf.control_dependencies([opt_c]):
        opt_c = tf.tuple(clipped_var_c)

    return opt_g, opt_c, c_loss, g_loss


def train():
    opt_g, opt_c, c_loss, g_loss = build_graph()

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    with tf.Session() as sess:

        # you need to initialize all variables
        tf.global_variables_initializer().run()

        if len(FLAGS.checkpoint_file) != 0:
            meta_file, ckpt_file = get_model_filenames(FLAGS.checkpoint_file)
            if os.path.exists(meta_file) and os.path.exists(ckpt_file + '.index'):
                set_A_vars = [v for v in tf.trainable_variables() if v.name.startswith('InceptionResnetV1')]
                pretrain_saver = tf.train.Saver(set_A_vars, max_to_keep=FLAGS.num_checkpoints)
                # pretrain_saver = tf.train.import_meta_graph(meta_file)
                pretrain_saver.restore(sess, ckpt_file)
                # saver.restore(sess, ckpt_file)
                print('restore from checkpoint: ', FLAGS.checkpoint_file)
            else:
                print('file not exists: ', meta_file, ckpt_file)
                sys.exit()

        train_imgs_batch = batch_iter(train_files, batch_size, num_epochs=1)

        train_batch_idx = 0
        for imgs_batch in train_imgs_batch:

            mask_np = np.random.binomial(1, 1 - FLAGS.corruption_level, imgs_batch.shape)
            train_batch_idx += 1

            if train_batch_idx % FLAGS.val_pre_train_batch_iter == 1:
                val_batch = batch_iter(val_files, batch_size, num_epochs=1, shuffle=True)
                val_loss = 0
                val_iter_count = 0

                for val_b in val_batch:
                    if val_iter_count > 10: break
                    val_iter_count += 1
                    val_mask_np = np.random.binomial(1, 1 - FLAGS.corruption_level, val_b.shape)

                    val_loss += sess.run(tf.reduce_mean(c_loss), feed_dict={input_image: val_b, mask: val_mask_np})

                val_loss /= val_iter_count
                print('iter: %6d, val_loss: %0.5f' % (train_batch_idx, val_loss))
                pass

            gl_m = 0
            for i in range(FLAGS.run_generator_per_train_batch_idx):
                _, gl = sess.run([opt_g, tf.reduce_mean(g_loss)], feed_dict={input_image: imgs_batch, mask: mask_np})
                gl_m += gl
            gl_m /= FLAGS.run_generator_per_train_batch_idx


            cl_m = 0
            for i in range(FLAGS.run_discriminator_per_train_batch_idx):
                _, cl = sess.run([opt_c, tf.reduce_mean(c_loss)], feed_dict={input_image: imgs_batch, mask: mask_np})
                cl_m += cl
            cl_m /= FLAGS.run_discriminator_per_train_batch_idx

            print('iter: %5d, g_loss: %0.5f, c_loss: %0.5f' % (train_batch_idx, gl_m, cl_m))

            pass
        pass




if __name__ == '__main__':
    train()
