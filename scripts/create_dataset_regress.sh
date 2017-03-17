#!/usr/bin/env sh

FILENAME_DIR=/home/jing/Documents/TableTennis/TableTennis/build
OUTPUT_DIR=/home/jing/Documents/CNN_tabletennis/data
CAFFE_ROOT=/home/jing/caffe/build_cmake

echo "Creating data lmdb..."

$CAFFE_ROOT/tools/convert_imageset / $FILENAME_DIR/filename_reg.txt $OUTPUT_DIR/regress_train_data_lmdb
$CAFFE_ROOT/tools/convert_imageset / $FILENAME_DIR/filename_reg_test.txt $OUTPUT_DIR/regress_val_data_lmdb

python scripts/gen_label.py --file=$FILENAME_DIR/label_reg.txt --outname=$OUTPUT_DIR/regress_train_label_lmdb
python scripts/gen_label.py --file=$FILENAME_DIR/label_reg_test.txt --outname=$OUTPUT_DIR/regress_val_label_lmdb
