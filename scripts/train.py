import sys
import os
import numpy as np
from pylab import *
import caffe

WEIGHTS_FILE = '/home/jing/Downloads/py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel'

caffe.set_device(0)
caffe.set_mode_gpu()

# Helper function for deprocessing preprocessed images, e.g., for display.
def deprocess_net_image(image, mean):
	image = image.copy()              # don't modify destructively
	image = image[::-1]               # BGR -> RGB
	image = image.transpose(1, 2, 0)  # CHW -> HWC
	image += mean          # (approximately) undo mean subtraction

	# clamp values in [0, 255]
	image[image < 0], image[image > 255] = 0, 255

	# round and cast from float32 to uint8
	image = np.round(image)
	image = np.require(image, dtype=np.uint8)

	return image

