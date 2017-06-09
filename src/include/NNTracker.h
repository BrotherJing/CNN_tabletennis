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
	bool track_vec(Mat &frame, std::vector<Rect> &proposals, float *output_prob, Rect *output_bbox);
	bool track(Mat &frame, Rect *proposals, int num_proposals, float *output_prob, Rect *output_bbox);

	void setProbThresh(float thresh);
	void setProposalRange(int range);
	void setMaxProposals(int proposals);
	void fail();

	~NNTracker();
	
	int num_patches;
	Mat *patches, *bboxes, *probs;
	Rect *contexts;
private:
	static const double LOW_PASS_FILTER;

	Rect get_context(Rect &bbox, float padding_ratio=1.5);
	Rect get_safe_context(Mat &frame, Rect &bbox, float padding_ratio=1.5);
	void get_many_contexts(Mat &frame, Rect &bbox);
	bool _too_far_away(Rect &bbox, Rect &bbox_last);
	float _distance_factor(Rect &bbox, Rect &bbox_last);
	bool _track(int num_patches, float *output_prob, Rect *output_bbox);

	Classifier &classifier;
	int max_proposal_;
	float prob_thresh;
	int proposal_range;

	bool last_found;
	int num_frames_found;
	
	Rect bbox_last;
};

#endif