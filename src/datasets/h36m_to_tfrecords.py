""" Convert MPI_INF_3DHP to TFRecords """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, exists
from os import makedirs

import h5py
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from common import convert_to_example_wmosh_h36m, ImageCoder, resize_img

from metadata import load_h36m_metadata


metadata = load_h36m_metadata()

tf.app.flags.DEFINE_string('data_directory', '/hpdata/h36m-fetch/processed',
                           'data directory: top of mpi-inf-3dhp')
tf.app.flags.DEFINE_string('output_directory',
                           '/hpdata/tf_datasets/h36m/',
                           'Output data directory')

tf.app.flags.DEFINE_string('split', 'train', 'train or trainval')
tf.app.flags.DEFINE_integer('train_shards', 500,
                            'Number of shards in training TFRecord files.')

FLAGS = tf.app.flags.FLAGS
# Subjects to include when preprocessing
included_subjects = {
    'S1': 1,
    'S5': 5,
    'S6': 6,
    'S7': 7,
    'S8': 8,
    'S9': 9,
    'S11': 11,
}

# Sequences with known issues
blacklist = {
    ('S11', '2', '2', '54138969'),  # Video file is corrupted
}

# Camera Id dict
sequence_mappings = metadata.sequence_mappings
cam_dict = {}
cam_dict_id = 0
for cam in metadata.camera_ids:
    cam_dict[cam] = cam_dict_id
    cam_dict_id += 1


def sample_frames(gt3ds):
    use_these = np.zeros(gt3ds.shape[0], bool)
    # Always use_these first frame.
    use_these[0] = True
    prev_kp3d = gt3ds[0]
    for itr, kp3d in enumerate(gt3ds):
        if itr > 0:
            # Check if any joint moved more than 200mm.
            if not np.any(np.linalg.norm(prev_kp3d - kp3d, axis=1) >= 40):
                continue
        use_these[itr] = True
        prev_kp3d = kp3d

    return use_these


def get_all_data(base_dir, sub_id, act_id, subact_ids, cam_id):
    com_dir = join(sub_id, metadata.action_names[act_id] + '-' + subact_ids)
    img_dir = join(base_dir, com_dir, 'imageSequence', cam_id)
    anno_path = join(base_dir + '_annots', com_dir, cam_id, 'annot.h5')

    all_gt2ds, all_gt3ds, all_img_paths = [], [], []
    all_cams = []

    with h5py.File(anno_path, 'r') as F:
        gt2ds = F['pose/2d'][:]
        gt3ds = F['pose/3d'][:]
        gt3ds_univ = F['pose/3d-univ'][:]
        cam_intr = F['intrinsics/'+cam_id][:]
        cam_intr_univ = F['intrinsics-univ/'+cam_id][:] 
        camera = F['camera'][:]
        frames = F['frame'][:]

    base_path = join(img_dir, 'img_%06d.jpg')
    img_paths = [base_path % frame for frame in frames]

    if gt3ds.shape[0] != len(img_paths):
        print('Not same paths?')
        import ipdb
        ipdb.set_trace()

    return img_paths, gt2ds, gt3ds, gt3ds_univ, cam_intr, cam_intr_univ, camera



def add_to_tfrecord(im_path,
                    gt2d,
                    gt3d,
                    gt3ds_univ,
                    cam_intr,
                    cam_intr_univ,
                    cam,
                    coder,
                    writer,
                    model=None,
                    sub_path=None):
    """
    gt2d all: 32 x 2
    gt3d all: 32 x 3
    """
    # Read image
    if not exists(im_path):
        # print('!!--%s doesnt exist! Skipping..--!!' % im_path)
        return False

    with tf.gfile.FastGFile(im_path, 'rb') as f:
        image_data = f.read()
    image = coder.decode_jpeg(coder.png_to_jpeg(image_data))
    assert image.shape[2] == 3

    # All kps are visible in mpi_inf_3dhp.
    min_pt = np.min(gt2d, axis=0)
    max_pt = np.max(gt2d, axis=0)
    center = np.round((min_pt + max_pt) / 2.).astype(np.int)

    # Crop 300x300 around the center
    # margin = 150
    # start_pt = np.maximum(center_scaled - margin, 0).astype(int)
    # end_pt = (center_scaled + margin).astype(int)
    # end_pt[0] = min(end_pt[0], width)
    # end_pt[1] = min(end_pt[1], height)
    # image_scaled = image_scaled[start_pt[1]:end_pt[1], start_pt[0]:end_pt[
    #     0], :]
    # # Update others too.
    # joints_scaled[:, 0] -= start_pt[0]
    # joints_scaled[:, 1] -= start_pt[1]
    # center_scaled -= start_pt
    # # Update principal point:
    # cam_scaled[1] -= start_pt[0]
    # cam_scaled[2] -= start_pt[1]
    height, width = image.shape[:2]

    # # Fix units: mm -> meter
    # gt3d = gt3d / 1000.

    # Encode image:
    image_data = coder.encode_jpeg(image)
    label = np.vstack([gt2d.T, np.ones((1, gt2d.shape[0]))])
    # pose and shape is not existent.
    pose, shape = None, None
    example = convert_to_example_wmosh_h36m(
        image_data, im_path, height, width, label, center, gt3d,
        pose, shape, [1,1] , np.int64(0), gt3ds_univ, cam_intr, cam_intr_univ)
    writer.write(example.SerializeToString())

    return True


def save_to_tfrecord(out_name, img_paths, gt2ds, gt3ds, gt3ds_univ, cam_intr, cam_intr_univ, camera, num_shards):
    coder = ImageCoder()
    i = 0
    # Count on shards
    fidx = 0
    # Count failures
    num_bad = 0
    while i < len(img_paths):
        tf_filename = out_name % fidx
        print('Starting tfrecord file %s' % tf_filename)
        with tf.python_io.TFRecordWriter(tf_filename) as writer:
            j = 0
            while i < len(img_paths) and j < num_shards:
                if i % 100 == 0:
                    print('Reading img %d/%d' % (i, len(img_paths)))
                success = add_to_tfrecord(img_paths[i], gt2ds[i], gt3ds[i], gt3ds_univ[i], 
                                    cam_intr, cam_intr_univ, camera[i], coder, writer)

                print(img_paths[i])
                i += 1
                if success:
                    j += 1
                else:
                    num_bad += 1

        fidx += 1

    print('Done, wrote to %s, num skipped %d' % (out_name, num_bad))


def process_h36m_train(data_dir, out_dir, is_train=False):
    if is_train:
        out_dir = join(out_dir, 'train')
        print('train set!')

    if not exists(out_dir):
        makedirs(out_dir)

    num_shards = FLAGS.train_shards

    subactions = []

    for subject in included_subjects.keys():
        subactions += [
            (subject, action, subaction)
            for action, subaction in sequence_mappings[subject].keys()
            if int(action) > 1  # Exclude '_ALL'
        ]
    
    for subject, action, subaction in tqdm(subactions, ascii=True, leave=False):
        
        for camera in tqdm(metadata.camera_ids, ascii=True, leave=False):
            print(subject, action, subaction, camera)
            if (subject, action, subaction, camera) in blacklist:
                continue

            out_path = join(out_dir, '{}_{}_{}_cam{}_train_%04d.tfrecord'.format(subject,
                            action, subaction, cam_dict[camera]))
                
            img_paths, gt2ds, gt3ds, gt3ds_univ, cam_intr, cam_intr_univ, camera = get_all_data(
                    data_dir, subject, action, subaction, camera)
    
            save_to_tfrecord(out_path, img_paths, gt2ds, gt3ds, gt3ds_univ, cam_intr, cam_intr_univ, camera, 
                         num_shards)


def main(unused_argv):
    print('Saving results to %s' % FLAGS.output_directory)

    if not exists(FLAGS.output_directory):
        makedirs(FLAGS.output_directory)

    if FLAGS.split == 'train' or FLAGS.split == 'trainval':
        is_train = FLAGS.split == 'train'
        process_h36m_train(
            FLAGS.data_directory, FLAGS.output_directory, is_train=is_train)
    else:
        print('Unknown split %s' % FLAGS.split)
        import ipdb
        ipdb.set_trace()


if __name__ == '__main__':
    tf.app.run()
