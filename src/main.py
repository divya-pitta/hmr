""" Driver for train """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .config import get_config, prepare_dirs, save_config
from .data_loader import DataLoader
from .trainer import HMRTrainer

import pdb

def main(config):
    prepare_dirs(config)

    # Load data on CPU
    with tf.device("/cpu:0"):
        data_loader = DataLoader(config)
        # 2D Images with 2d keypoints, and if present 3d SMPL keypoints too
        image_loader = data_loader.load()
        # 3D human meshes in terms of pose and shape
        smpl_loader = data_loader.get_smpl_loader()
    #pdb.set_trace()
    trainer = HMRTrainer(config, image_loader, smpl_loader)
    save_config(config)
    if config.num_gpus==1:
        trainer.train()
    else:
        trainer.train_multigpu()

if __name__ == '__main__':
    config = get_config()
    main(config)
