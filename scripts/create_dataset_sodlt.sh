#!/usr/bin/env sh

FILENAME_DIR=/home/jing/Documents/TableTennis/TableTennis/build
OUTPUT_DIR=/home/jing/Documents/CNN_tabletennis/data
CAFFE_ROOT=/home/jing/caffe/build_cmake

echo "Creating data lmdb..."

$CAFFE_ROOT/tools/convert_imageset / $FILENAME_DIR/train.txt $OUTPUT_DIR/train_data_lmdb
$CAFFE_ROOT/tools/convert_imageset / $FILENAME_DIR/val.txt $OUTPUT_DIR/val_data_lmdb

$CAFFE_ROOT/tools/convert_imageset / $FILENAME_DIR/train.label.txt $OUTPUT_DIR/train_label_lmdb --gray=true
$CAFFE_ROOT/tools/convert_imageset / $FILENAME_DIR/val.label.txt $OUTPUT_DIR/val_label_lmdb --gray=true

echo "Computing image mean..."

$CAFFE_ROOT/tools/compute_image_mean $OUTPUT_DIR/train_data_lmdb $OUTPUT_DIR/mean.binaryproto

python scripts/gen_py_mean.py
