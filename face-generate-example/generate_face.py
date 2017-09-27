import tensorflow as tf
import numpy as np
import os

from inception_resnet_v1 import inception_resnet_v1, inference
from utils import *
from dense_net import *
from generator_net import get_feature_to_image_net

tf.flags.DEFINE_float("lr", 0.001, "learning rate (default: 0.001)")
tf.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("checkpoint_file", "", "model restore")
tf.flags.DEFINE_string("pretrain_file", "", "pretrain model")
tf.flags.DEFINE_integer("val_pre_train_batch_iter", 50, "val model")
tf.flags.DEFINE_float("corruption_level", 0.2, "corruption_level")
tf.flags.DEFINE_float("val_ratio", 0.05, "val data ratio")
tf.flags.DEFINE_integer("run_discriminator_per_train_batch_idx", 2, "")
tf.flags.DEFINE_float("gan_coef", 0.02, "gan loss coef")
tf.flags.DEFINE_integer("batch_size", 64, "batch size")
tf.flags.DEFINE_integer("total_epoch", 100, "total epoch number")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

x_h = 160
x_w = 160
channels = 3
batch_size = FLAGS.batch_size

image_folder = '/home/blue/data/img_align_celeba_10000/'
# image_folder = '/home/mawei/MsCelebV1-Faces-Aligned_160/'
train_files, val_files = load_train_val_data(image_folder, FLAGS.val_ratio)

base_output_size = 96

scale_step = 6

d_net_base_output_size = 32
d_weights = {}
# 160  80  40  20  10    5
#  32  64 128 256 512 1024
with tf.device('gpu:0'), tf.name_scope("d_net"):
    for i in range(scale_step):
        scale = 2 ** i
        input_channel_size = int(d_net_base_output_size * scale / 2 if i > 0 else channels)
        output_channel_size = int(d_net_base_output_size * scale)
        shape = [3, 3, input_channel_size, output_channel_size]
        d_weights['conv%d' % (i+1)] = init_weights(shape)
        d_weights['bias%d' % (i+1)] = init_weights([output_channel_size])
    d_weights['fc_w'] = init_weights([d_net_base_output_size * (2 ** (scale_step-1)), 1])
    d_weights['fc_b'] = init_weights([1])

# print('d_weights:')
# for (k,v) in d_weights.items():
#     print(k, v)



# net define
with tf.device('gpu:0'):
    X = tf.placeholder("float", [None, x_h, x_w, channels])
    mask = tf.placeholder("float", [None, x_h, x_w, channels], name='mask')
    p_keep_conv = tf.placeholder("float")

def model(X, mask, p_keep_conv, batch_size):
    with tf.device('gpu:0'):

        tilde_X = mask * X  # corrupted X

        # f_net = vgg_a(tilde_X)
        # ir_net, _= inception_resnet_v1(tilde_X, is_training=False)
        ir_net, _= inference(tilde_X, 0.8, phase_train=False)

        net = get_feature_to_image_net(ir_net, batch_size, scale_step, base_output_size, channels, p_keep_conv, x_w, x_h)

        # use dense net
        # densenet = DenseNet(growth_rate=24, layers_per_block=5, keep_prob=0.8, is_training=True)
        #    c  96 192 384 768 1536
        # net = densenet.upsample_net(up_sacle_net, 
        #     [   5,  10,  20,  40, 80, 160], 
        #     [1536, 768, 384, 192, 96,   3],
        #     batch_size)

    return net, ir_net

def D(X, weights=d_weights):
    net = X
    for i in range(scale_step):
        net = tf.nn.conv2d(net, weights['conv%d' % (i+1)], strides=[1,1,1,1], padding="SAME")
        net = tf.nn.bias_add(net, weights['bias%d' % (i+1)])
        net = tf.nn.relu(net)

        if i < (scale_step - 1):
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pass
    ksize = np.ceil( x_w / (2 ** (scale_step-1)) )
    net = tf.nn.avg_pool(net, ksize=[1, ksize, ksize, 1], strides=[1,2,2,1], padding='SAME')

    fc_size = net.get_shape().as_list()[-1]
    net = tf.reshape(net, [-1, fc_size])
    net = tf.matmul(net, weights['fc_w']) + weights['fc_b']
    net = tf.nn.sigmoid(net)

    return net

Z, ir_net = model(X, mask, p_keep_conv, batch_size)
predict_op = Z 

l2_obj = tf.reduce_mean(tf.pow(tf.reshape(X, [batch_size, -1]) - tf.reshape(Z, [batch_size, -1]), 2))  # minimize squared error

dout_real = D(X)
dout_fake = D(Z)

G_obj = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=dout_fake, labels=tf.ones_like(dout_fake)))
D_obj_real = tf.reduce_mean( # use single side smoothing
        tf.nn.sigmoid_cross_entropy_with_logits(logits=dout_real, labels=(tf.ones_like(dout_real)-0.1))) 
D_obj_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=dout_fake, labels=tf.zeros_like(dout_fake))) 
D_obj = D_obj_real + D_obj_fake

D_opt = tf.train.AdamOptimizer().minimize(D_obj, var_list=d_weights.values())

train_opt = l2_obj + FLAGS.gan_coef * G_obj
train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(train_opt)  # construct an optimizer

# saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
trainable_variables = tf.trainable_variables()
saver = tf.train.Saver(trainable_variables, max_to_keep=FLAGS.num_checkpoints)

print('----------trainable vars------------')
for op in trainable_variables:
    print(op.name)
print('----------trainable vars------------')

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True
# Launch the graph in a session
with tf.Session(config=config) as sess:

    # you need to initialize all variables
    tf.global_variables_initializer().run()

    pretrain_vars = [v for v in tf.trainable_variables() if v.name.startswith('InceptionResnetV1')]
    restore_network(saver, sess, FLAGS.checkpoint_file, FLAGS.num_checkpoints, FLAGS.pretrain_file, pretrain_vars)

    min_loss = 1e20
    train_loss = 1e20
    train_batch_idx = 0
    for i in range(FLAGS.total_epoch):
        print('epoch: ', i)

        # train
        train_batch = batch_iter(train_files, batch_size, num_epochs=1)
        for batch in train_batch:
            # print('real batch size: %d' % len(batch))

            mask_np = np.random.binomial(1, 1 - FLAGS.corruption_level, batch.shape)

            # print(sess.run(ir_net, feed_dict={X: batch, mask: mask_np, p_keep_conv: 0.8}))

            train_batch_idx += 1

            if (train_batch_idx % FLAGS.val_pre_train_batch_iter) == 1:
                # test
                val_batch = batch_iter(val_files, batch_size, num_epochs=1, shuffle=True)
                val_loss = 0
                val_iter_count = 0
                discriminator_score = 0

                for val_b in val_batch:

                    # print('real val batch size: %d' % len(val_b))
                    if val_iter_count > 10: break
                    val_iter_count += 1
                    val_mask_np = np.random.binomial(1, 1 - FLAGS.corruption_level, val_b.shape)
                    l, s = sess.run([train_opt, tf.reduce_mean(dout_real)], feed_dict={X: val_b, mask: val_mask_np, p_keep_conv: 1.0})
                    val_loss += l
                    discriminator_score += s
                    pass
                val_loss /= val_iter_count
                discriminator_score /= val_iter_count

                print('batch iter: %10d, train loss: %.5f, val loss: %.5f, discriminator score: %.5f' % (train_batch_idx, train_loss, val_loss, discriminator_score))

                if val_loss < min_loss:
                    min_loss = val_loss
                    path = saver.save(sess, "models/model", global_step=train_batch_idx)
                    print("Saved model checkpoint to {}".format(path))

                    predicted_imgs = sess.run(predict_op, feed_dict={X: batch, mask: mask_np, p_keep_conv: 1.0})

                    vis(predicted_imgs, 'pred_%d' % train_batch_idx)
                    vis(np.array(batch, dtype=np.float32), 'in_%d' % train_batch_idx)

                    pass
                pass           

            _, train_loss, predicted_imgs = sess.run([train_op, train_opt, predict_op], feed_dict={X: batch, mask: mask_np, p_keep_conv: 0.8})               

            if train_batch_idx % FLAGS.run_discriminator_per_train_batch_idx == 0:
                sess.run(D_opt, feed_dict={X: batch, Z: predicted_imgs})
        pass

    # save the predictions for 300 images
    test_image_size = 128
    val_batch = batch_iter(val_files, test_image_size, num_epochs=1)
    for batch in val_batch:
        mask_np = np.random.binomial(1, 1 - FLAGS.corruption_level, batch.shape)
        predicted_imgs = sess.run(predict_op, feed_dict={X: batch, mask: mask_np, p_keep_conv: 1.0})
        input_imgs = batch
        vis(predicted_imgs,'pred')
        vis(input_imgs,'in')
        break
    print('Done')
