#!/usr/bin/env sh

FILENAME_DIR=/home/jing/Documents/TableTennis/TableTennis/build
OUTPUT_DIR=/home/jing/Documents/CNN_tabletennis/data
CAFFE_ROOT=/home/jing/caffe/build_cmake

echo "Creating data lmdb..."

$CAFFE_ROOT/tools/convert_imageset / $FILENAME_DIR/filename.txt $OUTPUT_DIR/train_data_lmdb
$CAFFE_ROOT/tools/convert_imageset / $FILENAME_DIR/filename_test.txt $OUTPUT_DIR/val_data_lmdb

echo "Computing image mean..."

$CAFFE_ROOT/tools/compute_image_mean $OUTPUT_DIR/train_data_lmdb $OUTPUT_DIR/mean.binaryproto

python scripts/gen_py_mean.py

python scripts/gen_label.py --file=$FILENAME_DIR/label.txt --outname=$OUTPUT_DIR/train_label_lmdb
python scripts/gen_label.py --file=$FILENAME_DIR/label_test.txt --outname=$OUTPUT_DIR/val_label_lmdb
