#ifndef HEADER_NNTRACKER
#define HEADER_NNTRACKER

#include "Classifier.h"

using namespace cv;

class NNTracker
{
public:
	NNTracker(Classifier &classifier);

	Rect running_avg(Rect last, Rect current);
	Rect get_bbox(Mat &prediction, Rect &context);
	bool track(Mat &frame, Rect *proposals, int num_proposals, float *output_prob, Rect *output_bbox);

	~NNTracker();
	
private:
	static const double LOW_PASS_FILTER;

	Rect get_context(Rect &bbox, float padding_ratio=1.0);

	Classifier &classifier;
	int max_proposal_;

	Mat *patches, *bboxes, *probs;
	Rect *contexts;
	Rect bbox_last;
};

#endif