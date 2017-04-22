/*
input:
- deploy prototxt
- weights
- mean file
- video to track
- initial bounding box(1*4 CvMat xml)
*/

#include <cstdio>
#include <opencv2/opencv.hpp>

#include "main.h"
#include "Classifier.h"
#include "NNTracker.h"

const double LOW_PASS_FILTER = 0.9;

/*struct timeval{
	long tv_sec;
	long tv_usec;
};*/

Rect run_avg(Rect last, Rect current){
	int new_width = int(last.width*LOW_PASS_FILTER + current.width*(1 - LOW_PASS_FILTER));
	int new_height = int(last.height*LOW_PASS_FILTER + current.height*(1 - LOW_PASS_FILTER));
	//int new_width = last.width;
	//int new_height = last.height;
	int center_x = current.x + current.width/2;
	int center_y = current.y + current.height/2;
	return Rect(center_x - new_width/2, center_y - new_height/2, new_width, new_height);
}

Rect get_bbox(Mat prediction, Rect context){
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

int track(VideoCapture &video, CvMat *init_bbox_mat, Classifier &classifier){

	Mat frame;

	//the bounding box of the object in every frame
	Rect bbox = Rect(CV_MAT_ELEM(*init_bbox_mat, int, 0, 0),
		CV_MAT_ELEM(*init_bbox_mat, int, 0, 1),
		CV_MAT_ELEM(*init_bbox_mat, int, 0, 2),
		CV_MAT_ELEM(*init_bbox_mat, int, 0, 3));

	int longer_edge = bbox.width>bbox.height?bbox.width:bbox.height;
	int padding_left = longer_edge*1.;
	int padding_top = longer_edge*1.;

	//the area to crop from the frame
	Rect context = Rect(bbox.x - padding_left,
		bbox.y - padding_top,
		bbox.width + padding_left*2,
		bbox.height + padding_top*2);

	//input image to the network
	//Mat patch(Size(100, 100), frame.type());

#ifdef DEBUG_MODE
	namedWindow("display", WINDOW_AUTOSIZE);
	namedWindow("patch", WINDOW_AUTOSIZE);
	namedWindow("prob_map", WINDOW_AUTOSIZE);
#endif

	struct timeval t1,t2;
	double timeuse;
	while(true){

#ifdef TIMING
		gettimeofday(&t1,NULL);
#endif
		
		video >> frame;
		if(frame.empty())return 0;

		//crop the context in last frame
		Mat patch = frame(context);
		//cv::resize(roi, patch, patch.size());

#ifdef TRACK_DEBUG_MODE
		/*rectangle(frame, Point(bbox.x, bbox.y),
			Point(bbox.x+bbox.width, bbox.y+bbox.height), CV_RGB(0x66, 0x00, 0x00), 1, CV_AA);
		rectangle(frame, Point(context.x, context.y),
			Point(context.x+context.width, context.y+context.height), CV_RGB(0x66, 0x00, 0x00), 1, CV_AA);*/
		imshow("display", frame);
		waitKey(0);
#endif

		Mat imgs[1], probs[1], bboxes[1];
		imgs[0] = patch;
		printf("context: %d %d %d %d\n", context.x, context.y, context.width, context.height);
		classifier.PredictN(imgs, 1, probs, bboxes);
		Rect bbox_proposal = get_bbox(bboxes[0], context);
		bbox = run_avg(bbox, bbox_proposal);

		//update context
		longer_edge = bbox.width>bbox.height?bbox.width:bbox.height;
		padding_left = longer_edge*1.;
		padding_top = longer_edge*1.;
		context.x = bbox.x - padding_left;
		context.y = bbox.y - padding_top;
		context.width = bbox.width + padding_left*2;
		context.height = bbox.height + padding_top*2;

#ifdef TRACK_DEBUG_MODE
		printf("probability %f\n", probs[0].at<float>(0,0));
		printf("bbox: %d %d %d %d\n", bbox.x, bbox.y, bbox.width, bbox.height);
		rectangle(frame, Point(bbox.x, bbox.y),
			Point(bbox.x+bbox.width, bbox.y+bbox.height), CV_RGB(0xff, 0x00, 0x00), 1, CV_AA);
		/*rectangle(frame, Point(context.x, context.y),
			Point(context.x+context.width, context.y+context.height), CV_RGB(0xff, 0x00, 0x00), 1, CV_AA);*/
		imshow("display", frame);
		imshow("patch", patch);
		char c = waitKey(0);
		if(c=='s'){
			imwrite("patch.jpg", patch);
		}else if(c==27)break;
#endif

#ifdef TIMING
		gettimeofday(&t2,NULL);
		timeuse = (t2.tv_sec - t1.tv_sec)*1000.0 + (t2.tv_usec - t1.tv_usec)/1000.0;
		printf("Use Time:%fms\n",timeuse);
#endif
	}

#ifdef TRACK_DEBUG_MODE
	destroyWindow("display");
	destroyWindow("patch");
	destroyWindow("prob_map");
#endif
}

int main(int argc, char **argv){
	string proto_path, weights_path, mean_path, video_path, init_bbox_path;
	proto_path = (argc>1)?argv[1]:"models/sodlt_deploy.prototxt";
	weights_path = (argc>2)?argv[2]:"example_sodlt_iter_10000.caffemodel";
	mean_path = (argc>3)?argv[3]:"data/mean.binaryproto";
	video_path = (argc>4)?argv[4]:"0001R.MP4";
	init_bbox_path = (argc>5)?argv[5]:"init_bbox.xml";

	Classifier classifier(proto_path, weights_path, mean_path);

	VideoCapture video(video_path);

	CvMat *init_bbox = (CvMat*)cvLoad(init_bbox_path.c_str());

	//track(video, init_bbox, classifier);

	//use the NNTracker class
	NNTracker tracker(classifier);
	Rect proposals[1];
	proposals[0] = Rect(CV_MAT_ELEM(*init_bbox, int, 0, 0),
		CV_MAT_ELEM(*init_bbox, int, 0, 1),
		CV_MAT_ELEM(*init_bbox, int, 0, 2),
		CV_MAT_ELEM(*init_bbox, int, 0, 3));
	float prob;
	Rect bbox = Rect(0,0,0,0);

	struct timeval t1,t2;
	double timeuse;
	namedWindow("display", WINDOW_AUTOSIZE);
	while(true){
		Mat frame;
		video>>frame;
		if(frame.empty())break;
#ifdef TIMING
		gettimeofday(&t1,NULL);
#endif
		tracker.track(frame, proposals, 1, &prob, &bbox);
#ifdef TIMING
		gettimeofday(&t2,NULL);
		timeuse = (t2.tv_sec - t1.tv_sec)*1000.0 + (t2.tv_usec - t1.tv_usec)/1000.0;
		printf("Use Time:%fms\n",timeuse);
#endif

#ifdef TRACK_DEBUG_MODE
		rectangle(frame, Point(bbox.x, bbox.y),
			Point(bbox.x+bbox.width, bbox.y+bbox.height), CV_RGB(0xff, 0x00, 0x00), 1, CV_AA);
		imshow("display", frame);
		char c = waitKey(0);
		if(c==27)break;
#endif
	}
	destroyWindow("display");

	return 0;
}