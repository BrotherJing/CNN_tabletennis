#!/usr/bin/env sh

#VGG_MODEL_DIR=/home/jing/Downloads/py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel
MODEL_DIR=/home/jing/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
CAFFE_ROOT=/home/jing/caffe/build_cmake

$CAFFE_ROOT/tools/caffe train --solver=models/solver.prototxt --weights=$MODEL_DIR --gpu=0