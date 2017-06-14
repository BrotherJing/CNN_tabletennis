#!/usr/bin/env sh

FILENAME_DIR=/home/jing/Documents/TableTennis/TableTennis/build
OUTPUT_DIR=/home/jing/Documents/CNN_tabletennis/data
CAFFE_ROOT=/home/jing/caffe/build_cmake

echo "Creating dataset lmdb for SO-CNN..."

#format of train.txt(or val.txt):
# [absolute path to the image patch] [class label(0 or 1)]

echo "Training data..."
$CAFFE_ROOT/tools/convert_imageset / $FILENAME_DIR/train.txt $OUTPUT_DIR/train_data_lmdb

echo "Validation data..."
$CAFFE_ROOT/tools/convert_imageset / $FILENAME_DIR/val.txt $OUTPUT_DIR/val_data_lmdb

echo "Creating probability map lmdb for SO-CNN"

#format of xx.label.txt:
# [absolute path to the probability map] [class label(ignored)]

echo "Training label..."
$CAFFE_ROOT/tools/convert_imageset / $FILENAME_DIR/train.label.txt $OUTPUT_DIR/train_label_lmdb --gray=true

echo "Validation label..."
$CAFFE_ROOT/tools/convert_imageset / $FILENAME_DIR/val.label.txt $OUTPUT_DIR/val_label_lmdb --gray=true

echo "Creating dataset lmdb for regression layer..."

cp $FILENAME_DIR/train.txt $FILENAME_DIR/train.regress.txt
sed -i '/.*0$/d' $FILENAME_DIR/train.regress.txt
sed -i '/^0 0 0 0$/d' $FILENAME_DIR/train.bbox.txt
cp $FILENAME_DIR/val.txt $FILENAME_DIR/val.regress.txt
sed -i '/.*0$/d' $FILENAME_DIR/val.regress.txt
sed -i '/^0 0 0 0$/d' $FILENAME_DIR/val.bbox.txt

#format of xx.regress.txt:
# [absolute path to the image patch] 1

echo "Training data..."
$CAFFE_ROOT/tools/convert_imageset / $FILENAME_DIR/train.regress.txt $OUTPUT_DIR/train_regress_data_lmdb

echo "Validation data..."
$CAFFE_ROOT/tools/convert_imageset / $FILENAME_DIR/val.regress.txt $OUTPUT_DIR/val_regress_data_lmdb

echo "Creating bounding box label lmdb..."

#format of xx.bbox.txt:
# x y w h

echo "Training label..."
python scripts/gen_label.py --file=$FILENAME_DIR/train.bbox.txt --outname=$OUTPUT_DIR/train_regress_label_lmdb

echo "Validation label..."
python scripts/gen_label.py --file=$FILENAME_DIR/val.bbox.txt --outname=$OUTPUT_DIR/val_regress_label_lmdb

echo "Computing image mean..."

$CAFFE_ROOT/tools/compute_image_mean $OUTPUT_DIR/train_data_lmdb $OUTPUT_DIR/mean.binaryproto

#python scripts/gen_py_mean.py

echo "Done."