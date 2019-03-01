from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
from glob import glob

import tensorflow as tf
import sys
import argparse

from .datasets.common import read_images_from_tfrecords
from .datasets.common import bytes_feature
from .util import data_utils
from .tf_smpl.batch_lbs import batch_rodrigues
from .benchmark import evaluate_h36m

# Generate new tfrecords which have the paired images and paired keypoints

def generate_tfrecord_paired(filenames, inputfile_dir, outfile_dir, skip_value=20, num_shards=500, has_3d=False):
    global sess
    with tf.name_scope(None, 'read_data', filenames):
        for fname in filenames:
            filename = join(inputfile_dir, fname + '.tfrecord')
            print("File being converted to joint tfrecord: ", filename)
            raw_dataset1 = tf.data.TFRecordDataset([filename])
            raw_dataset2 = tf.data.TFRecordDataset([filename])
            raw_dataset2 = raw_dataset2.skip(skip_value)
            iterator1 = raw_dataset1.make_one_shot_iterator()
            iterator2 = raw_dataset2.make_one_shot_iterator()
	    next_pair1 = iterator1.get_next()
	    next_pair2 = iterator2.get_next()
            train_out = join(outfile_dir, fname + '_%03d.tfrecord')
            fidx = 0
            i = 0
            while True:
                try:
                    tf_filename = train_out % fidx
                    print('Starting tfrecord file %s' % tf_filename)
                    with tf.python_io.TFRecordWriter(tf_filename) as writer:
                        j = 0
                        while j < num_shards:
                            if i % 100 == 0:
                                print('Converting image %d' % (i))
                            pair1 = sess.run(next_pair1)
			    pair2 = sess.run(next_pair2)
                            example = tf.train.Example(features=tf.train.Features(feature={
                                'image1/tfrecord': bytes_feature(
                                    tf.compat.as_bytes(pair1)),
                                'image2/tfrecord': bytes_feature(
                                    tf.compat.as_bytes(pair2))
                            }))
                            writer.write(example.SerializeToString())
                            i += 1
                            j += 1
                    fidx += 1
                except tf.errors.OutOfRangeError:
                    break

all_seq_files = []
test_file_directory = '/hmr/data/test_tf_datasets/tf_records_human36m_wjoints/test'
paired_file_directory = '/hmr/data/test_tf_datasets/paired_h36m/train'

all_pairs, actions = evaluate_h36m.get_h36m_seqs(3)
print("Generated all the action sequences")

for itr, seq_info in enumerate(all_pairs):
    sub_id, action, trial_id, cam_id = seq_info
    file_seq_name = 'S%d_%s_%d_cam%01d' % (sub_id, action, trial_id, cam_id)
    print("The sequence name is ", file_seq_name)
    all_seq_files.append(file_seq_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-root_dir', type=str, required=True)

    opt = parser.parse_args()
    sess = None
    global sess
    if sess is None:
        sess = tf.Session()
    generate_tfrecord_paired(all_seq_files, opt.root_dir+test_file_directory, opt.root_dir+paired_file_directory)

if __name__ == '__main__':
    main()
