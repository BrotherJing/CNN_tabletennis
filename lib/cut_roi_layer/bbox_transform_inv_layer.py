import caffe
import numpy as np

class BBoxTransformInvLayer(caffe.Layer):

	def setup(self, bottom, top):
		assert len(bottom) == 2, 'require rois and transforms'
		#assert len(top) == 2, 'require a bbox top and a prob top'

	def reshape(self, bottom, top):
		top[0].reshape(*bottom[0].data.shape)

	def forward(self, bottom, top):

		boxes = bottom[0].data
		deltas = bottom[1].data

		widths = boxes[:, 2] - boxes[:, 0] + 1.0
		heights = boxes[:, 3] - boxes[:, 1] + 1.0
		ctr_x = boxes[:, 0] + 0.5 * widths
		ctr_y = boxes[:, 1] + 0.5 * heights

		dx = deltas[:, 0::4]
		dy = deltas[:, 1::4]
		dw = deltas[:, 2::4]
		dh = deltas[:, 3::4]

		pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
		pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
		pred_w = np.exp(dw) * widths[:, np.newaxis]
		pred_h = np.exp(dh) * heights[:, np.newaxis]

		pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
		# x1
		pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
		# y1
		pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
		# x2
		pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
		# y2
		pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

		top[0].data[...] = pred_boxes

	def backward(self, top, propagate_down, bottom):
		pass

