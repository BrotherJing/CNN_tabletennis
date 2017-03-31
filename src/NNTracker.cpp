#include <opencv2/opencv.hpp>
#include <cstdio>

#include "NNTracker.h"
#include "Classifier.h"

#define TIMING

NNTracker::NNTracker(Classifier &classifier):
	classifier(classifier){
	max_proposal_ = 4;
	patches = new Mat[max_proposal_];
	bboxes = new Mat[max_proposal_];
	probs = new Mat[max_proposal_];

	contexts = new Rect[max_proposal_];

	bbox_last = Rect(0,0,0,0);
}

NNTracker::~NNTracker(){
	delete [] patches;
	delete [] bboxes;
	delete [] probs;
	delete [] contexts;
}

Rect NNTracker::running_avg(Rect last, Rect current){
	int new_width = int(last.width*LOW_PASS_FILTER + current.width*(1 - LOW_PASS_FILTER));
	int new_height = int(last.height*LOW_PASS_FILTER + current.height*(1 - LOW_PASS_FILTER));
	int center_x = current.x + current.width/2;
	int center_y = current.y + current.height/2;
	return Rect(center_x - new_width/2, center_y - new_height/2, new_width, new_height);
}

Rect NNTracker::get_bbox(Mat &prediction, Rect &context){
	Rect bbox;
	bbox = Rect(prediction.at<float>(0,0),
		prediction.at<float>(0,1),
		prediction.at<float>(0,2) - prediction.at<float>(0,0),
		prediction.at<float>(0,3) - prediction.at<float>(0,1));
	double ratio_w = 2*context.width*1.0/100;
	double ratio_h = 2*context.height*1.0/100;
	bbox.x = context.x + bbox.x * ratio_w;
	bbox.y = context.y + bbox.y * ratio_h;
	bbox.width = bbox.width * ratio_w;
	bbox.height = bbox.height * ratio_h;
	return bbox;
}

Rect NNTracker::get_context(Rect &bbox, float padding_ratio){
	int longer_edge = bbox.width>bbox.height?bbox.width:bbox.height;
	int padding_left = longer_edge*padding_ratio;
	int padding_top = longer_edge*padding_ratio;

	//the area to crop from the frame
	return Rect(bbox.x - padding_left,
		bbox.y - padding_top,
		bbox.width + padding_left*2,
		bbox.height + padding_top*2);
}

bool NNTracker::track(Mat &frame, Rect *proposals, int num_proposals, float *output_prob, Rect *output_bbox){

	//we want to add the context from last frame to be a proposal..
	num_proposals = (num_proposals>max_proposal_-1)?max_proposal_-1:num_proposals;

	for(int i=0;i<num_proposals;++i){
		contexts[i] = get_context(proposals[i], 0.6);
		//input image to the network
		patches[i] = frame(contexts[i]);
	}
	if(bbox_last.width!=0){
		contexts[num_proposals] = get_context(bbox_last);
		patches[num_proposals] = frame(contexts[num_proposals]);
		num_proposals++;
	}

#ifdef TIMING
	struct timeval t1,t2;
	double timeuse;
	gettimeofday(&t1,NULL);
#endif

	classifier.PredictN(patches, num_proposals, probs, bboxes);

	//find the best one
	float max_prob = 0;
	int max_prob_idx = 0;
	for(int i=0;i<num_proposals;++i){
		if(probs[i].at<float>(0,0)>max_prob){
			max_prob = probs[i].at<float>(0,0);
			max_prob_idx = i;
		}
	}
	Rect bbox_rect = get_bbox(bboxes[max_prob_idx], contexts[max_prob_idx]);
	if(bbox_last.width==0){
		//new target
		bbox_last = bbox_rect;
	}else{
		bbox_last = running_avg(bbox_last, bbox_rect);
	}
	*output_prob = max_prob;
	*output_bbox = bbox_last;

#ifdef TIMING
	gettimeofday(&t2,NULL);
	timeuse = (t2.tv_sec - t1.tv_sec)*1000.0 + (t2.tv_usec - t1.tv_usec)/1000.0;
	printf("Use Time:%fms\n",timeuse);
#endif

	return true;
}