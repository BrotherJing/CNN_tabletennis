import caffe
import numpy as np

class CutROILayer(caffe.Layer):

	def setup(self, bottom, top):
		assert len(bottom) == 1, 'require single layer bottom'
		assert bottom[0].data.ndim == 4, 'require a probability map'
		#assert len(top) == 2, 'require a bbox top and a prob top'

	def reshape(self, bottom, top):
		top[0].reshape(*(bottom[0].data.shape[0], 4))
		if len(top)>1:
			top[1].reshape(bottom[0].data.shape[0])

	def forward(self, bottom, top):
		prob_max = np.max(bottom[0].data, axis=(1,2,3))#max per batch
		if len(top)>1:
			top[1].data[...] = prob_max
		thresh = prob_max/2
		max_of_each_row = np.max(bottom[0].data, axis=-1)#batch * 1 * h
		max_of_each_col = np.max(bottom[0].data, axis=-2)#batch * 1 * w
		#max_of_each_row[max_of_each_row < thresh] = 0
		#max_of_each_col[max_of_each_col < thresh] = 0
		for b in range(max_of_each_row.shape[0]):
			for i in range(max_of_each_row.shape[2]):
				if max_of_each_row[b,0,i]>thresh[b]:
					if i==0 or max_of_each_row[b,0,i-1]<=thresh[b]:
						top[0].data[b,1] = i#top
					if i==max_of_each_row.shape[2]-1 or max_of_each_row[b,0,i+1]<=thresh[b]:
						top[0].data[b,3] = i+1#bottom
			for i in range(max_of_each_col.shape[2]):
				if max_of_each_col[b,0,i]>thresh[b]:
					if i==0 or max_of_each_col[b,0,i-1]<=thresh[b]:
						top[0].data[b,0] = i#left
					if i==max_of_each_col.shape[2]-1 or max_of_each_col[b,0,i+1]<=thresh[b]:
						top[0].data[b,2] = i+1#right
	
	def backward(self, top, propagate_down, bottom):
		pass

