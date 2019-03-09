"""
TF util operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from demo import get_silhouette

# Assuming the silhouette is of the exact same size as the image in input!!!!! Else this will fail.
# def silhouette_l1_loss(sil_gt, joints, vertices, cams, name=None):
#     #TODO: Make this faster using batch operations!!!
#     with tf.name_scope(name, "keypoint_l1_loss", [sil_gt, joints, vertices, cams]):
#         sils = []
#         for index in range(sil_gt.shape[0]):
#             inv_sil = get_silhouette(
#                 # sil_gt[index],
#                 joints[index],
#                 vertices[index],
#                 cams[index]
#             )
#             # Inverting 0s and 1s
#             sil = 1 / (2e-5 + inv_sil)
#             #Normalizing to between 0 and 1
#             sil = (sil - np.min(sil))/(np.max(sil) - np.min(sil))
#             sils.append(sil)
#         res = tf.losses.absolute_difference(sil_gt, sils)
#         return res

def silhouette_l1_loss(sil_gt, sil, name=None):
    with tf.name_scope(name, "keypoint_l1_loss", [sil_gt, sil]):
        sil_gt = tf.reshape(sil_gt, (-1,3))
        sil = tf.reshape(sil, (-1,3))
        res = -1*tf.losses.absolute_difference(sil_gt, sil)
        return res

def keypoint_l1_loss(kp_gt, kp_pred, scale=1., name=None):
    """
    computes: \Sum_i [0.5 * vis[i] * |kp_gt[i] - kp_pred[i]|] / (|vis|)
    Inputs:
      kp_gt  : N x K x 3
      kp_pred: N x K x 2
    """
    with tf.name_scope(name, "keypoint_l1_loss", [kp_gt, kp_pred]):
        kp_gt = tf.reshape(kp_gt, (-1, 3))
        kp_pred = tf.reshape(kp_pred, (-1, 2))

        vis = tf.expand_dims(tf.cast(kp_gt[:, 2], tf.float32), 1)
        res = tf.losses.absolute_difference(kp_gt[:, :2], kp_pred, weights=vis)
        return res


def compute_3d_loss(params_pred, params_gt, has_gt3d):
    """
    Computes the l2 loss between 3D params pred and gt for those data that has_gt3d is True.
    Parameters to compute loss over:
    3Djoints: 14*3 = 42
    rotations:(24*9)= 216
    shape: 10
    total input: 226 (gt SMPL params) or 42 (just joints)

    Inputs:
      params_pred: N x {226, 42}
      params_gt: N x {226, 42}
      # has_gt3d: (N,) bool
      has_gt3d: N x 1 tf.float32 of {0., 1.}
    """
    with tf.name_scope("3d_loss", [params_pred, params_gt, has_gt3d]):
        weights = tf.expand_dims(tf.cast(has_gt3d, tf.float32), 1)
        res = tf.losses.mean_squared_error(
            params_gt, params_pred, weights=weights) * 0.5
        return res


def align_by_pelvis(joints):
    """
    Assumes joints is N x 14 x 3 in LSP order.
    Then hips are: [3, 2]
    Takes mid point of these points, then subtracts it.
    """
    with tf.name_scope("align_by_pelvis", [joints]):
        left_id = 3
        right_id = 2
        pelvis = (joints[:, left_id, :] + joints[:, right_id, :]) / 2.
        return joints - tf.expand_dims(pelvis, axis=1)
