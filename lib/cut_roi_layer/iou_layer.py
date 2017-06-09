import caffe
import numpy as np
import hashlib
import cv2

class IOULayer(caffe.Layer):

	def setup(self, bottom, top):
		assert len(bottom) >= 2, 'require two layer bottom: bbox_pred, bbox_gt'
		#assert bottom[0].data.ndim == 4, 'require a probability map'
		#assert len(top) == 2, 'require a bbox top and a prob top'

	def reshape(self, bottom, top):
		top[0].reshape(1)

	def forward(self, bottom, top):
		
		iou = 0.
		for i in range(bottom[0].data.shape[0]):
			this_iou = self.iou(bottom[0].data[i]*2, bottom[1].data[i])#predicted bbox is 50*50!
			#print this_iou
			iou += this_iou
		iou /= bottom[0].data.shape[0]

		top[0].data[0] = iou

		#if len(bottom)>2:
			#print bottom[2].data[0].shape
			#print bottom[2].data[0,0,0]
			#hs = hashlib.md5(bottom[2].data[0]).hexdigest()
			#print hs
			#grad_blob = bottom[2].data[0].transpose((1,2,0))    # c01 -> 01c
			#grad_img = grad_blob[:, :,[2,1,0]]  # e.g. BGR -> RGB
			#grad_img = (grad_img-grad_img.min())/(grad_img.max()-grad_img.min())
			#img_int = (grad_img*255).astype(np.uint8)
			#cv2.imwrite(hs+'.jpg', img_int)

	def backward(self, top, propagate_down, bottom):
		pass

	def iou(self, rect1, rect2):
		#print rect1, rect2
		x1 = rect1[0]
		y1 = rect1[1]
		w1 = rect1[2] - rect1[0]
		h1 = rect1[3] - rect1[1]
		
		x2 = rect2[0]
		y2 = rect2[1]
		w2 = rect2[2] - rect2[0]
		h2 = rect2[3] - rect2[1]

		endx = max(x1+w1, x2+w2)
		startx = min(x1, x2)
		w = w1+w2-(endx-startx)
		endy = max(y1+h1, y2+h2)
		starty = min(y1, y2)
		h = h1+h2-(endy-starty)

		if w==0 or h==0:
			return 0

		area = w*h
		area1 = w1*h1
		area2 = w2*h2
		return area*1.0/(area1+area2-area)