# -*- coding: utf-8 -*-
"""
* @File Name:   		utils.py
* @Author:				Wang Yang
* @Created Date:		2017-08-19 09:08:47
* @Last Modified Data:	2017-09-26 21:51:14
* @Desc:					
*
"""
import cPickle as pickle

import matplotlib # to plot images
# Force matplotlib to not use any X-server backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
import re
import numpy as np
import random
import cv2
import tensorflow as tf

def save_var(var_name, var, temp_folder):
    if False == os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    var_file_path = '%s/%s.pkl' %(temp_folder, var_name)
    fp = open(var_file_path, 'w+')
    pickle.dump(var, fp, -1)
    fp.close()
    print('save_var: ' + var_file_path)
    return var_file_path

def load_var(var_name, temp_folder):
    var_file_path = '%s/%s.pkl' %(temp_folder, var_name)
    if False == os.path.exists(var_file_path):
        print('var file path not exist: ' + var_file_path)
        return None

    fp = open(var_file_path)
    var = pickle.load(fp)
    fp.close()
    print('load_var: ' + var_file_path)
    return var

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

def batch_iter(data, batch_size, num_epochs, resize_size=(160, 160), shuffle=False):
    data_size = len(data)

    num_batches_per_epoch = int((len(data) - 1) / batch_size)

    if (shuffle):
        random.shuffle(data)

    for epoch in range(num_epochs):

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            batch = []
            for d in data[start_index:end_index]:
                try:
                    img = cv2.resize(cv2.imread(d), resize_size)  
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

def restore_network(saver, sess, checkpoint_file, num_checkpoints, pretrain_file, pretrain_vars):
    if len(checkpoint_file) != 0:
        meta_file, ckpt_file = get_model_filenames(checkpoint_file)
        if os.path.exists(meta_file) and os.path.exists(ckpt_file + '.index'):
            # pretrain_saver = tf.train.import_meta_graph(meta_file)
            # pretrain_saver.restore(sess, ckpt_file)
            saver.restore(sess, ckpt_file)
            print('restore from checkpoint: ', checkpoint_file)
        else:
            print('file not exists: ', meta_file, ckpt_file)
            sys.exit()
    elif pretrain_file != '':
        meta_file, ckpt_file = get_model_filenames(pretrain_file)
        if os.path.exists(meta_file) and os.path.exists(ckpt_file + '.index'):
            pretrain_saver = tf.train.Saver(pretrain_vars, max_to_keep=num_checkpoints)
            pretrain_saver.restore(sess, ckpt_file)
            print('restore from pretrain: ', pretrain_file)
        else:
            print('file not exists: ', meta_file, ckpt_file)
            sys.exit()
        pass
    pass

def load_train_val_data(image_folder, val_ratio):
    img_files = load_var('img_files', temp_folder='temps')
    if img_files == None:
        img_files = []
        for parent, folders, filenames in os.walk(image_folder):
            for filename in filenames:
                line = parent + '/' + filename
                #print(line)
                img_files.append(line)
            pass
        save_var('img_files', img_files, temp_folder='temps')

    print('total image file: %d' % len(img_files))
    val_count = int(len(img_files) * val_ratio) 
    train_files = img_files[:-val_count]
    val_files = img_files[-val_count:]

    return train_files, val_files