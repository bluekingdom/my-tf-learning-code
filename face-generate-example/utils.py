# -*- coding: utf-8 -*-
"""
* @File Name:   		utils.py
* @Author:				Wang Yang
* @Created Date:		2017-08-19 09:08:47
* @Last Modified Data:	2017-08-19 09:11:39
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