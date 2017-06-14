#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cmath>

#include "NNTracker.h"
#include "Classifier.h"

const double NNTracker::LOW_PASS_FILTER = 0.85;

NNTracker::NNTracker(Classifier &classifier):
	classifier(classifier){
	max_proposal_ = 4;
	prob_thresh = 0.0f;
	proposal_range = -1;

	last_found = false;
	num_frames_found = 0;

	patches = new Mat[max_proposal_];
	bboxes = new Mat[max_proposal_];
	probs = new Mat[max_proposal_];

	contexts = new Rect[max_proposal_];
	bbox_last = Rect(0,0,0,0);
}

NNTracker::~NNTracker(){
}

void NNTracker::setMaxProposals(int proposals){
	max_proposal_ = proposals;
	patches = new Mat[max_proposal_];
	bboxes = new Mat[max_proposal_];
	probs = new Mat[max_proposal_];
	contexts = new Rect[max_proposal_];
}

void NNTracker::fail(){
	last_found = false;
	num_frames_found = 0;
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
	int x,y;
	int diff = bbox.width - bbox.height;
	int longer_edge = diff>0?bbox.width:bbox.height;
	int padding = longer_edge*padding_ratio;
	if(diff>0){
		x = bbox.x;
		y = bbox.y - diff/2;
	}else{
		y = bbox.y;
		x = bbox.x - diff/2;
	}
	//the area to crop from the frame
	return Rect(x - padding,
		y - padding,
		longer_edge + padding*2,
		longer_edge + padding*2);
}

Rect NNTracker::get_safe_context(Mat &frame, Rect &bbox, float padding_ratio){
	int x,y;
	int diff = bbox.width - bbox.height;
	int longer_edge = diff>0?bbox.width:bbox.height;
	int padding = longer_edge*padding_ratio;
	int padding_left = padding, padding_right = padding, padding_top = padding, padding_bottom = padding;
	if(diff>0){
		x = bbox.x;
		y = bbox.y - diff/2;
	}else{
		y = bbox.y;
		x = bbox.x - diff/2;
	}
	if(x - padding_left < 0){
		padding_right += (padding_left - x);
		padding_left = x;
	}
	if(y - padding_top < 0){
		padding_bottom += (padding_top - y);
		padding_top = y;
	}
	if(x + longer_edge + padding_right > frame.cols){
		padding_left += padding_right - (frame.cols - (x + longer_edge));
		padding_right = frame.cols - (x + longer_edge);
	}
	if(y + longer_edge + padding_bottom > frame.rows){
		padding_top += padding_bottom - (frame.rows - (y + longer_edge));
		padding_bottom = frame.rows - (y + longer_edge);
	}
	//the area to crop from the frame
	return Rect(x - padding_left,
		y - padding_top,
		longer_edge + padding_left + padding_right,
		longer_edge + padding_top + padding_bottom);
}

void NNTracker::get_many_contexts(Mat &frame, Rect &bbox){
	Rect context;
	int center_x, center_y;
	for(int i=0;i<2;++i){
		for(int j=0;j<2;++j){
			if(num_patches>=max_proposal_)break;
			//center_x = bbox.x - bbox.width/2 + i * bbox.width * 2;
			//center_y = bbox.y - bbox.height/2 + j * bbox.height * 2;
			center_x = bbox.x + i * bbox.width;
			center_y = bbox.y + j * bbox.height;
			if(center_x<0||center_x>=frame.cols||
				center_y<0||center_y>=frame.rows)continue;
			context.x = center_x - bbox.width*3/2;
			context.y = center_y - bbox.height*3/2;
			context.width = bbox.width * 3;
			context.height = bbox.height * 3;
			contexts[num_patches] = get_safe_context(frame, context, 0.0);
			patches[num_patches] = frame(contexts[num_patches]);
			num_patches++;
		}
	}
}

bool NNTracker::track_vec(Mat &frame, std::vector<Rect> &proposals, float *output_prob, Rect *output_bbox){
	num_patches = 0;

	if(last_found){
		if(num_frames_found>10){
			get_many_contexts(frame, bbox_last);
		}
		else{
			contexts[num_patches] = get_safe_context(frame, bbox_last, 2);
			if(contexts[num_patches].x<0||contexts[num_patches].y<0||
				contexts[num_patches].x+contexts[num_patches].width>frame.cols||
				contexts[num_patches].y+contexts[num_patches].height>frame.rows||
				contexts[num_patches].width<0||contexts[num_patches].height<0){
			}else{
				patches[num_patches] = frame(contexts[num_patches]);
				num_patches++;
			}
		}
	}
	for(int i=0;i<proposals.size();++i){
		if(num_patches>=max_proposal_)break;
		contexts[num_patches] = get_safe_context(frame, proposals[i], 0.6);
		if(contexts[num_patches].x<0||contexts[num_patches].y<0||
			contexts[num_patches].x+contexts[num_patches].width>frame.cols||
			contexts[num_patches].y+contexts[num_patches].height>frame.rows||
			contexts[num_patches].width<0||contexts[num_patches].height<0){
			continue;
		}
		if(last_found && proposal_range>0){
			if(_too_far_away(contexts[num_patches], bbox_last))continue;
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
	num_patches = 0;

	if(last_found){
		if(num_frames_found>10){
			get_many_contexts(frame, bbox_last);
		}else{
			contexts[num_patches] = get_safe_context(frame, bbox_last, 1);
			if(contexts[num_patches].x<0||contexts[num_patches].y<0||
				contexts[num_patches].x+contexts[num_patches].width>frame.cols||
				contexts[num_patches].y+contexts[num_patches].height>frame.rows||
				contexts[num_patches].width<0||contexts[num_patches].height<0){
			}else{
				patches[num_patches] = frame(contexts[num_patches]);
				num_patches++;
			}
		}
	}
	for(int i=0;i<num_proposals;++i){
		if(num_patches>=max_proposal_)break;
		contexts[num_patches] = get_safe_context(frame, proposals[i], 0.6);
		if(contexts[num_patches].x<0||contexts[num_patches].y<0||
			contexts[num_patches].x+contexts[num_patches].width>frame.cols||
			contexts[num_patches].y+contexts[num_patches].height>frame.rows||
			contexts[num_patches].width<0||contexts[num_patches].height<0){
			continue;
		}
		if(last_found && proposal_range>0){
			/*if(contexts[num_patches].x<bbox_last.x - proposal_range||
				contexts[num_patches].x>bbox_last.x + proposal_range||
				contexts[num_patches].y<bbox_last.y - proposal_range||
				contexts[num_patches].y>bbox_last.y + proposal_range){
				continue;
			}*/
			if(_too_far_away(contexts[num_patches], bbox_last))continue;
		}
		patches[num_patches] = frame(contexts[num_patches]);
		num_patches++;
		//input image to the network
	}
	return _track(num_patches, output_prob, output_bbox);
}

bool NNTracker::_too_far_away(Rect &bbox, Rect &bbox_last){
	if(proposal_range<=0)return false;
	int dis_x = (bbox.x+bbox.width/2) - (bbox_last.x+bbox_last.width/2);
	int dis_y = (bbox.y+bbox.height/2) - (bbox_last.y+bbox_last.height/2);
	float distance = dis_x*dis_x+dis_y*dis_y;
	return distance > proposal_range*proposal_range;
}

float NNTracker::_distance_factor(Rect &bbox, Rect &bbox_last){
	if(proposal_range<=0)return 1;
	int dis_x = (bbox.x+bbox.width/2) - (bbox_last.x+bbox_last.width/2);
	int dis_y = (bbox.y+bbox.height/2) - (bbox_last.y+bbox_last.height/2);
	float distance = sqrt(dis_x*dis_x+dis_y*dis_y);
	float reg = proposal_range*proposal_range/10;
	float result = exp(-distance/reg);
	printf("distance=%f, factor=%f\n", distance, result);
	return result;
}

bool NNTracker::_track(int num_patches, float *output_prob, Rect *output_bbox){

	if(num_patches==0){
		last_found = false;
		num_frames_found = 0;
		return false;
	}

	classifier.PredictN(patches, num_patches, probs, bboxes);

	//find the best one
	float max_prob = 0;
	int max_prob_idx = 0;
	for(int i=0;i<num_patches;++i){
		float prob = probs[i].at<float>(0,0);
		/*if(last_found){
			Rect bbox_rect = get_bbox(bboxes[i], contexts[i]);
			prob *= _distance_factor(bbox_rect, bbox_last);
		}*/
		if(prob>max_prob){
			max_prob = prob;
			max_prob_idx = i;
		}
	}
	//max_prob = probs[max_prob_idx].at<float>(0,0);//restore original probability
	if(max_prob < prob_thresh){
		last_found = false;
		num_frames_found = 0;
		return false;
	}
	Rect bbox_rect = get_bbox(bboxes[max_prob_idx], contexts[max_prob_idx]);
	if(!last_found
		||max_prob>0.95
		//||_too_far_away(bbox_rect, bbox_last)
		){
		//new target
		bbox_last = bbox_rect;
		last_found = true;
		num_frames_found++;
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