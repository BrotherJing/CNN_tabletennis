import sys
import numpy as np
import matplotlib.pyplot as plt
import caffe

PATCH_SIZE = 227

WEIGHTS_FILE = '/home/jing/Documents/CNN_tabletennis/example_iter_3323.caffemodel'
DEPLOY_FILE = '/home/jing/Documents/CNN_tabletennis/models/deploy.prototxt'
MEAN_FILE = '/home/jing/Documents/CNN_tabletennis/data/mean.npy'

mu = np.load(MEAN_FILE)
mu = mu.mean(1).mean(1)

net = caffe.Net(DEPLOY_FILE, WEIGHTS_FILE, caffe.TEST)

transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))#nChannels w h?
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))#RGB to BGR

img_name = sys.argv[1]

#batch_size = net.blobs['data'].data.shape[0]

image = caffe.io.load_image(img_name)
plt.imshow(image)
transformed_image = transformer.preprocess('data', image)
net.blobs['data'].data[0,...] = transformed_image

output = net.forward()

#class_pred = output['my_cls_score'][0]
class_pred = output['my_fc8'][0]
bbox_pred = output['my_pred'][0]

print 'predicted class is ', class_pred.argmax()
print 'class score: ', class_pred
print 'bbox is ', bbox_pred*PATCH_SIZE