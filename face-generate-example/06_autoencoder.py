import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import os
import re

import matplotlib # to plot images
# Force matplotlib to not use any X-server backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from inception_resnet_v1 import inception_resnet_v1

tf.flags.DEFINE_float("lr", 0.001, "learning rate (default: 0.001)")
tf.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("checkpoint_file", "", "model restore")
tf.flags.DEFINE_integer("val_pre_train_batch_iter", 10, "val model")
tf.flags.DEFINE_float("corruption_level", 0.0, "corruption_level")
tf.flags.DEFINE_float("val_ratio", 0.05, "val data ratio")
tf.flags.DEFINE_integer("run_discriminator_per_train_batch_idx", 2, "")
tf.flags.DEFINE_float("gan_coef", 0.2, "gan loss coef")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def get_model_filenames(checkpoint_file):
    prefix = checkpoint_file[checkpoint_file.rfind('/')+1:]
    if os.path.exists(checkpoint_file + '.meta') and os.path.exists(checkpoint_file + '.data-00000-of-00001'):
        return checkpoint_file + '.meta', checkpoint_file

    model_dir = checkpoint_file[:checkpoint_file.rfind('/')]
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('can not find meta file for the checkpoint file (%s)' % checkpoint_file)

    # meta_files = [s for s in files if '.ckpt' in s]
    meta_file = meta_files[0]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]

    #     step_str = re.match(r'(^model-(\d+).data-[\d\-\w]+)', f)
    #     if step_str is not None and len(step_str.groups())>=2:
    #         step = int(step_str.groups()[1])
    #         if step > max_step:
    #             max_step = step
    #             ckpt_file = step_str.groups()[0]

    return os.path.join(model_dir, meta_file), os.path.join(model_dir, ckpt_file)


## Visualizing reconstructions
def vis(images, save_name):
    dim = images.shape[0]
    n_image_rows = int(np.ceil(np.sqrt(dim)))
    n_image_cols = int(np.ceil(dim * 1.0/n_image_rows))
    gs = gridspec.GridSpec(n_image_rows,n_image_cols,top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)
    for g,count in zip(gs,range(int(dim))):
        ax = plt.subplot(g)
        ax.imshow(images[count])
        ax.set_xticks([])
        ax.set_yticks([])
    save_path = './result_imgs/'
    if False == os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig('./result_imgs/' + save_name + '_vis.png')

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

x_h = 160
x_w = 160
channels = 3

def batch_iter(data, batch_size, num_epochs):
    data_size = len(data)

    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    for epoch in range(num_epochs):

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            batch = []
            for d in data[start_index:end_index]:
                try:
                    img = cv2.resize(cv2.imread(d), (x_h, x_w))  
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype(np.float32) / 255.0
                    batch.append(img) 
                except Exception as err:
                    print(err) 
                    # batch = [cv2.resize(cv2.imread(d), (x_h, x_w)) for d in data[start_index:end_index]]

            if len(batch) == 0:
                continue

            yield np.array(batch)
        pass
    pass



image_folder = '/home/blue/data/img_align_celeba/'
# image_folder = '/home/mawei/MsCelebV1-Faces-Aligned_160/'
img_files = []
for parent, folders, filenames in os.walk(image_folder):
    for filename in filenames:
        line = parent + '/' + filename
        #print(line)
        img_files.append(line)
    pass

print('total image file: %d' % len(img_files))
val_count = int(len(img_files) * FLAGS.val_ratio) 
train_files = img_files[:-val_count]
val_files = img_files[-val_count:]

base_output_size = 96

scale_step = 6
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

print('weights:')
for (k,v) in weights.items():
    print(k, v)

d_net_base_output_size = 32
d_weights = {}
# 160  80  40  20  10    5
#  32  64 128 256 512 1024
for i in range(scale_step):
    scale = 2 ** i
    input_channel_size = int(d_net_base_output_size * scale / 2 if i > 0 else channels)
    output_channel_size = int(d_net_base_output_size * scale)
    shape = [3, 3, input_channel_size, output_channel_size]
    d_weights['conv%d' % (i+1)] = init_weights(shape)
    d_weights['bias%d' % (i+1)] = init_weights([output_channel_size])
d_weights['fc_w'] = init_weights([d_net_base_output_size * (2 ** (scale_step-1)), 1])
d_weights['fc_b'] = init_weights([1])

print('d_weights:')
for (k,v) in d_weights.items():
    print(k, v)

def get_generate_net(net, batch_size):
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

    return net

# net define
X = tf.placeholder("float", [None, x_h, x_w, channels])
mask = tf.placeholder("float", [None, x_h, x_w, channels], name='mask')
p_keep_conv = tf.placeholder("float")
batch_size = tf.shape(X)[0]

def model(X, mask, p_keep_conv, batch_size):

    tilde_X = mask * X  # corrupted X

    # f_net = vgg_a(tilde_X)
    _, end_points = inception_resnet_v1(tilde_X, is_training=False)
    f_net = end_points['Mixed_8b']

    f_net_shape = f_net.get_shape().as_list()
    print(f_net_shape)
    fc_shape = f_net_shape[1] * f_net_shape[2] * f_net_shape[3]
    feature1 = tf.reshape(f_net, [-1, fc_shape])   
    feature1 = tf.nn.dropout(feature1, p_keep_conv)
    feature2 = tf.contrib.layers.fully_connected(feature1, fc_shape, scope='feature2')

    up_sacle_net = tf.reshape(feature2, [-1, f_net_shape[1], f_net_shape[2], f_net_shape[3]])

    return get_generate_net(up_sacle_net, batch_size)

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

predict_op = Z = model(X, mask, p_keep_conv, batch_size)

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


# Launch the graph in a session
with tf.Session() as sess:

    batch_size = 64
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    if len(FLAGS.checkpoint_file) != 0:
        meta_file, ckpt_file = get_model_filenames(FLAGS.checkpoint_file)
        if os.path.exists(meta_file) and os.path.exists(ckpt_file + '.index'):
            # saver = tf.train.import_meta_graph(meta_file)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            saver._max_to_keep = 1
            saver.restore(sess, ckpt_file)
            print('restore from checkpoint: ', FLAGS.checkpoint_file)
        else:
            print('file not exists: ', meta_file, ckpt_file)
            sys.exit()
    else:
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        print('use new saver.')


    min_loss = 1e20
    train_loss = 1e20
    train_batch_idx = 0
    for i in range(100):
        print('epoch: ', i)

        # train
        train_batch = batch_iter(train_files, batch_size, num_epochs=1)
        for batch in train_batch:

            mask_np = np.random.binomial(1, 1 - FLAGS.corruption_level, batch.shape)

            train_batch_idx += 1

            if (train_batch_idx % FLAGS.val_pre_train_batch_iter) == 1:
                # test
                val_batch = batch_iter(val_files, batch_size, num_epochs=1)
                val_loss = 0
                val_iter_count = 0
                discriminator_score = 0

                for val_b in val_batch:
                    if val_iter_count > 50: break
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
    test_image_size = 64
    val_batch = batch_iter(val_files, test_image_size, num_epochs=1)
    for batch in val_batch:
        mask_np = np.random.binomial(1, 1 - FLAGS.corruption_level, batch.shape)
        predicted_imgs = sess.run(predict_op, feed_dict={X: batch, mask: mask_np, p_keep_conv: 1.0})
        input_imgs = batch
        vis(predicted_imgs,'pred')
        vis(input_imgs,'in')
        break
    print('Done')
