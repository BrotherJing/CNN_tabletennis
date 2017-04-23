#include <opencv2/opencv.hpp>
#include <cstdio>

#include "NNTracker.h"
#include "Classifier.h"

const double NNTracker::LOW_PASS_FILTER = 0.85;

NNTracker::NNTracker(Classifier &classifier):
	classifier(classifier){
	max_proposal_ = 4;
	prob_thresh = 0.0f;
	proposal_range = -1;

	last_found = false;

	patches = new Mat[max_proposal_];
	bboxes = new Mat[max_proposal_];
	probs = new Mat[max_proposal_];

	contexts = new Rect[max_proposal_];
	bbox_last = Rect(0,0,0,0);
}

NNTracker::~NNTracker(){
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

bool NNTracker::track_vec(Mat &frame, std::vector<Rect> &proposals, float *output_prob, Rect *output_bbox){
	int num_patches = 0;

	if(last_found){
		contexts[num_patches] = get_context(bbox_last);
		if(contexts[num_patches].x<0||contexts[num_patches].y<0||
			contexts[num_patches].x+contexts[num_patches].width>frame.cols||
			contexts[num_patches].y+contexts[num_patches].height>frame.rows){
		}else{
			patches[num_patches] = frame(contexts[num_patches]);
			num_patches++;
		}
	}
	for(int i=0;i<proposals.size();++i){
		if(num_patches>=max_proposal_)break;
		contexts[num_patches] = get_context(proposals[i], 0.6);
		if(contexts[num_patches].x<0||contexts[num_patches].y<0||
			contexts[num_patches].x+contexts[num_patches].width>frame.cols||
			contexts[num_patches].y+contexts[num_patches].height>frame.rows){
			continue;
		}
		if(last_found && proposal_range>0){
			if(contexts[num_patches].x<bbox_last.x - proposal_range||
				contexts[num_patches].x>bbox_last.x + proposal_range||
				contexts[num_patches].y<bbox_last.y - proposal_range||
				contexts[num_patches].y>bbox_last.y + proposal_range){
				continue;
			}
		}
		patches[num_patches] = frame(contexts[num_patches]);
		num_patches++;
		//input image to the network
	}
	return _track(num_patches, output_prob, output_bbox);
}

bool NNTracker::track(Mat &frame, Rect *proposals, int num_proposals, float *output_prob, Rect *output_bbox){

	//we want to add the context from last frame to be a proposal..
	//num_proposals = (num_proposals>max_proposal_-1)?max_proposal_-1:num_proposals;
	int num_patches = 0;

	if(last_found){
		contexts[num_patches] = get_context(bbox_last);
		if(contexts[num_patches].x<0||contexts[num_patches].y<0||
			contexts[num_patches].x+contexts[num_patches].width>frame.cols||
			contexts[num_patches].y+contexts[num_patches].height>frame.rows){
		}else{
			patches[num_patches] = frame(contexts[num_patches]);
			num_patches++;
		}
	}
	for(int i=0;i<num_proposals;++i){
		if(num_patches>=max_proposal_)break;
		contexts[num_patches] = get_context(proposals[i], 0.6);
		if(contexts[num_patches].x<0||contexts[num_patches].y<0||
			contexts[num_patches].x+contexts[num_patches].width>frame.cols||
			contexts[num_patches].y+contexts[num_patches].height>frame.rows){
			continue;
		}
		if(last_found && proposal_range>0){
			if(contexts[num_patches].x<bbox_last.x - proposal_range||
				contexts[num_patches].x>bbox_last.x + proposal_range||
				contexts[num_patches].y<bbox_last.y - proposal_range||
				contexts[num_patches].y>bbox_last.y + proposal_range){
				continue;
			}
		}
		patches[num_patches] = frame(contexts[num_patches]);
		num_patches++;
		//input image to the network
	}
	if(num_patches==0){
		last_found = false;
		return false;
	}

	classifier.PredictN(patches, num_patches, probs, bboxes);

	//find the best one
	float max_prob = 0;
	int max_prob_idx = 0;
	for(int i=0;i<num_patches;++i){
		if(probs[i].at<float>(0,0)>max_prob){
			max_prob = probs[i].at<float>(0,0);
			max_prob_idx = i;
		}
	}
	if(max_prob < prob_thresh){
		last_found = false;
		return false;
	}
	Rect bbox_rect = get_bbox(bboxes[max_prob_idx], contexts[max_prob_idx]);
	if(!last_found){
		//new target
		bbox_last = bbox_rect;
		last_found = true;
	}else{
		bbox_last = running_avg(bbox_last, bbox_rect);
	}
	*output_prob = max_prob;
	*output_bbox = bbox_last;

	return true;
}

bool NNTracker::_track(int num_patches, float *output_prob, Rect *output_bbox){

	if(num_patches==0){
		last_found = false;
		return false;
	}

	classifier.PredictN(patches, num_patches, probs, bboxes);

	//find the best one
	float max_prob = 0;
	int max_prob_idx = 0;
	for(int i=0;i<num_patches;++i){
		if(probs[i].at<float>(0,0)>max_prob){
			max_prob = probs[i].at<float>(0,0);
			max_prob_idx = i;
		}
	}
	if(max_prob < prob_thresh){
		last_found = false;
		return false;
	}
	Rect bbox_rect = get_bbox(bboxes[max_prob_idx], contexts[max_prob_idx]);
	if(!last_found){
		//new target
		bbox_last = bbox_rect;
		last_found = true;
	}else{
		bbox_last = running_avg(bbox_last, bbox_rect);
	}
	*output_prob = max_prob;
	*output_bbox = bbox_last;

	return true;
}

void NNTracker::setProbThresh(float thresh){
	prob_thresh = thresh;
}

void NNTracker::setProposalRange(int range){
	proposal_range = range;
}