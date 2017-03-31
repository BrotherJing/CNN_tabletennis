/*
input:
- deploy prototxt
- weights
- mean file
- video to track
- initial bounding box(1*4 CvMat xml)
*/

#include <caffe/caffe.hpp>
#include <cstdio>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>

#include "main.h"
#include "Classifier.h"

const double LOW_PASS_FILTER = 0.9;

/*struct timeval{
	long tv_sec;
	long tv_usec;
};*/

CvRect run_avg(CvRect last, CvRect current){
	int new_width = int(last.width*LOW_PASS_FILTER + current.width*(1 - LOW_PASS_FILTER));
	int new_height = int(last.height*LOW_PASS_FILTER + current.height*(1 - LOW_PASS_FILTER));
	//int new_width = last.width;
	//int new_height = last.height;
	int center_x = current.x + current.width/2;
	int center_y = current.y + current.height/2;
	return cvRect(center_x - new_width/2, center_y - new_height/2, new_width, new_height);
}

CvRect get_bbox(Mat prediction, CvRect context){
	CvRect bbox;
	bbox = cvRect(prediction.at<float>(0,0),
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

int track(CvCapture *video, CvMat *init_bbox_mat, Classifier &classifier){

	IplImage *frame = cvQueryFrame(video);
	if(!frame)return -1;

	//the bounding box of the object in every frame
	CvRect bbox = cvRect(CV_MAT_ELEM(*init_bbox_mat, int, 0, 0),
		CV_MAT_ELEM(*init_bbox_mat, int, 0, 1),
		CV_MAT_ELEM(*init_bbox_mat, int, 0, 2),
		CV_MAT_ELEM(*init_bbox_mat, int, 0, 3));

	int longer_edge = bbox.width>bbox.height?bbox.width:bbox.height;
	int padding_left = longer_edge*1.;
	int padding_top = longer_edge*1.;

	//the area to crop from the frame
	CvRect context = cvRect(bbox.x - padding_left,
		bbox.y - padding_top,
		bbox.width + padding_left*2,
		bbox.height + padding_top*2);

	//input image to the network
	IplImage *patch = cvCreateImage(cvSize(100, 100), frame->depth, frame->nChannels);

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
		
		frame = cvQueryFrame(video);
		if(!frame)return 0;

		//crop the context in last frame
		cvSetImageROI(frame, context);
		cvResize(frame, patch);
		cvResetImageROI(frame);

#ifdef DEBUG_MODE
		cvRectangle(frame, cvPoint(bbox.x, bbox.y),
			cvPoint(bbox.x+bbox.width, bbox.y+bbox.height), CV_RGB(0x66, 0x00, 0x00), 1, CV_AA);
		cvRectangle(frame, cvPoint(context.x, context.y),
			cvPoint(context.x+context.width, context.y+context.height), CV_RGB(0x66, 0x00, 0x00), 1, CV_AA);
		cvShowImage("display", frame);
		cvWaitKey(0);
#endif

		Mat patch_mat(patch);
		Mat imgs[1], probs[1], bboxes[1];
		imgs[0] = patch_mat;
		classifier.PredictN(imgs, 1, probs, bboxes);
		CvRect bbox_proposal = get_bbox(bboxes[0], context);
		//Mat prediction = classifier.Predict(patch_mat);
		//CvRect bbox_proposal = get_bbox(prediction, context);
		bbox = run_avg(bbox, bbox_proposal);

		//update context
		longer_edge = bbox.width>bbox.height?bbox.width:bbox.height;
		padding_left = longer_edge*1.;
		padding_top = longer_edge*1.;
		context.x = bbox.x - padding_left;
		context.y = bbox.y - padding_top;
		context.width = bbox.width + padding_left*2;
		context.height = bbox.height + padding_top*2;

#ifdef DEBUG_MODE
		printf("probability %f\n", probs[0].at<float>(0,0));
		printf("bbox %d %d %d %d\n", bbox.x, bbox.y, bbox.width, bbox.height);
		cvRectangle(frame, cvPoint(bbox.x, bbox.y),
			cvPoint(bbox.x+bbox.width, bbox.y+bbox.height), CV_RGB(0xff, 0x00, 0x00), 1, CV_AA);
		cvRectangle(frame, cvPoint(context.x, context.y),
			cvPoint(context.x+context.width, context.y+context.height), CV_RGB(0xff, 0x00, 0x00), 1, CV_AA);
		cvShowImage("display", frame);
		cvShowImage("patch", patch);
		char c = cvWaitKey(0);
		if(c=='s'){
			cvSaveImage("patch.jpg", patch);
		}else if(c==27)break;
#endif

#ifdef TIMING
		gettimeofday(&t2,NULL);
		timeuse = (t2.tv_sec - t1.tv_sec)*1000.0 + (t2.tv_usec - t1.tv_usec)/1000.0;
		printf("Use Time:%fms\n",timeuse);
#endif
	}

#ifdef DEBUG_MODE
	cvDestroyWindow("display");
	cvDestroyWindow("patch");
	cvDestroyWindow("prob_map");
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

	CvCapture *video = cvCreateFileCapture(video_path.c_str());

	CvMat *init_bbox = (CvMat*)cvLoad(init_bbox_path.c_str());

	track(video, init_bbox, classifier);

	return 0;
}