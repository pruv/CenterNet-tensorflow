from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tfrecord_voc_utils as voc_utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tfrecord = voc_utils.dataset2tfrecord('../VOC2007_Test/Annotations', '../VOC2007_Test/JPEGImages',
                                      '../data_test/', 'test', 10)
print(tfrecord)
