# -*- coding: utf-8 -*-
"""
* @File Name:           random_change_face.py
* @Author:              Wang Yang
* @Created Date:        2017-10-18 20:24:49
* @Last Modified Data:  2017-10-18 21:33:24
* @Desc:                    
*
"""

from inception_resnet_v1 import inception_resnet_v1, inference
from utils import *
from generator_net import get_feature_to_image_net
from tqdm import tqdm

pretrain_file = 'models/feature_model-99'
data_root = '/home/blue/data/img_align_celeba_10000/'
data_save_root = '/home/blue/data/img_align_celeba_10000_reproduce/'

reproduce_count = 2

x_h = 160
x_w = 160
channels = 3
batch_size = 1
base_output_size = 96
scale_step = 6
corruption_level = 0.2

def load_file(data_root):
    all_files = []
    for parent, folders, filenames in os.walk(data_root):
        for filename in filenames:
            line = parent + '/' + filename
            if '_' in line[line.rindex('/'):]:
                os.remove(line)
                print('pass file: %s' % line)
                continue
            if cv2.imread(line) is None:
                print('load file fail: %s' % line)
                os.remove(line)
                continue
            all_files.append(line)

    if len(all_files) == 0:
        print('no file in data_root: %s' % data_root)
        sys.exit()

    return all_files

def model(X, mask, p_keep_conv, batch_size):
    tilde_X = mask * X  # corrupted X

    ir_net, _= inference(tilde_X, 0.8, phase_train=False)

    net = get_feature_to_image_net(ir_net, batch_size, scale_step, base_output_size, channels, p_keep_conv, x_w, x_h)

    return net, ir_net


def main():

    all_files = load_file(data_root)

    X = tf.placeholder("float", [None, x_h, x_w, channels]) 
    mask = tf.placeholder("float", [None, x_h, x_w, channels], name='mask')
    p_keep_conv = tf.placeholder("float")

    predict, feature = model(X, mask, p_keep_conv, batch_size)

    with tf.Session() as sess:

        tf.global_variables_initializer().run()

        pretrain_vars = [v for v in tf.trainable_variables() if v.name.startswith('InceptionResnetV1')]
        pretrain_vars += [v for v in tf.trainable_variables() if v.name.startswith('generate_net')]
        for op in pretrain_vars:
            print(op.name)
        restore_network(None, sess, '', 1, pretrain_file, pretrain_vars)


        for file in tqdm(all_files):
            image = cv2.imread(file)
            try:
                image = cv2.resize(image, (x_w, x_h))
                image = np.array([image])
            except:
                continue

            for i in range(reproduce_count):
                mask_np = np.random.binomial(1, 1 - corruption_level, image.shape)
                predicted_img = sess.run(predict, feed_dict={X: image, mask: mask_np, p_keep_conv: 0.7})
                save_path = '%s_%d.jpg' % (data_save_root + file[file.rindex('/')+1:file.rindex('.')], i)
                cv2.imwrite(save_path, predicted_img[0, :, :, :])


if __name__ == '__main__':
    main()
