import caffe
import numpy as np

class BBoxTransformLayer(caffe.Layer):

	def setup(self, bottom, top):
		assert len(bottom) == 2, 'require rois and ground truth'
		#assert len(top) == 2, 'require a bbox top and a prob top'

	def reshape(self, bottom, top):
		top[0].reshape(*bottom[0].data.shape)

	def forward(self, bottom, top):
		# Proposal ROIs (x1, y1, x2, y2) coming from RPN
		# (i.e., rpn.proposal_layer.ProposalLayer), or any other source
		all_rois = bottom[0].data
		# GT boxes (x1, y1, x2, y2)
		# TODO(rbg): it's annoying that sometimes I have extra info before
		# and other times after box coordinates -- normalize to one format
		gt_rois = bottom[1].data.reshape(all_rois.shape)

		ex_widths = all_rois[:, 2] - all_rois[:, 0] + 1.0
		ex_heights = all_rois[:, 3] - all_rois[:, 1] + 1.0
		ex_ctr_x = all_rois[:, 0] + 0.5 * ex_widths
		ex_ctr_y = all_rois[:, 1] + 0.5 * ex_heights

		gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
		gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
		gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
		gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

		gt_widths = gt_widths/ 2.0 #100 to 50
		gt_heights = gt_heights/ 2.0
		gt_ctr_x = gt_ctr_x/ 2.0
		gt_ctr_y = gt_ctr_y/ 2.0

		targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
		targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
		targets_dw = np.log(gt_widths / ex_widths)
		targets_dh = np.log(gt_heights / ex_heights)

		top[0].data[...] = np.vstack(
			(targets_dx, targets_dy, targets_dw, targets_dh)).transpose()

	def backward(self, top, propagate_down, bottom):
		pass

