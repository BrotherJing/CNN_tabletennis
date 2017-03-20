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
#include "Classifier.h"

const double LOW_PASS_FILTER = 0.9;

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
	IplImage prob_map = prediction;

	static CvMemStorage *mem_storage = NULL;
	static IplImage *map_copy = NULL;
	static CvSeq *contours;
	if(mem_storage==NULL){
		map_copy = cvCreateImage(cvGetSize(&prob_map), IPL_DEPTH_8U, 1);
		mem_storage = cvCreateMemStorage(0);
	}else{
		cvClearMemStorage(mem_storage);
	}
	CvRect bbox, mid_bbox;
	CvPoint center;
	double max_prob, min_prob;
	cvMinMaxLoc(&prob_map, &min_prob, &max_prob);
	cout<<"max prob:"<<max_prob<<endl;
	for(int i=1;i<4;++i){
		double t = max_prob*i/4;
		cvThreshold(&prob_map, map_copy, t, 255, CV_THRESH_BINARY);
		cvFindContours(map_copy, mem_storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		if(contours!=NULL){
			bbox = cvBoundingRect(contours);
			if(i==3)mid_bbox = bbox;
		}
	}
	center = cvPoint(bbox.x+bbox.width/2, bbox.y+bbox.height/2);
	cvCircle(&prob_map, center, 1, CV_RGB(0xff, 0xff, 0xff), 1);
	bbox = cvRect(center.x - mid_bbox.width/2,
		center.y - mid_bbox.height/2,
		mid_bbox.width,
		mid_bbox.height);
	cvShowImage("prob_map", &prob_map);
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
	CvRect bbox = cvRect(CV_MAT_ELEM(*init_bbox_mat, int, 0, 0),
		CV_MAT_ELEM(*init_bbox_mat, int, 0, 1),
		CV_MAT_ELEM(*init_bbox_mat, int, 0, 2),
		CV_MAT_ELEM(*init_bbox_mat, int, 0, 3));
	int longer_edge = bbox.width>bbox.height?bbox.width:bbox.height;
	int padding_left = longer_edge*1.5;
	int padding_top = longer_edge*1.5;
	CvRect context = cvRect(bbox.x - padding_left,
		bbox.y - padding_top,
		bbox.width + padding_left*2,
		bbox.height + padding_top*2);
	IplImage *patch = cvCreateImage(cvSize(100, 100), frame->depth, frame->nChannels);
	cvSetImageROI(frame, context);
	cvResize(frame, patch);
	cvResetImageROI(frame);

	namedWindow("display", WINDOW_AUTOSIZE);
	namedWindow("patch", WINDOW_AUTOSIZE);
	namedWindow("prob_map", WINDOW_AUTOSIZE);
	while(true){
		frame = cvQueryFrame(video);
		if(!frame)return 0;
		cvSetImageROI(frame, context);
		cvResize(frame, patch);
		cvResetImageROI(frame);

		cvRectangle(frame, cvPoint(bbox.x, bbox.y),
			cvPoint(bbox.x+bbox.width, bbox.y+bbox.height), CV_RGB(0x66, 0x00, 0x00), 1, CV_AA);
		cvRectangle(frame, cvPoint(context.x, context.y),
			cvPoint(context.x+context.width, context.y+context.height), CV_RGB(0x66, 0x00, 0x00), 1, CV_AA);
		cvShowImage("display", frame);
		cvWaitKey(0);

		Mat patch_mat(patch);
		Mat prediction = classifier.Predict(patch_mat);
		CvRect new_bbox = get_bbox(prediction, context);
		bbox = run_avg(bbox, new_bbox);
		longer_edge = bbox.width>bbox.height?bbox.width:bbox.height;
		padding_left = longer_edge*1.5;
		padding_top = longer_edge*1.5;
		context.x = bbox.x - padding_left;
		context.y = bbox.y - padding_top;
		context.width = bbox.width + padding_left*2;
		context.height = bbox.height + padding_top*2;
		
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
	}
	cvDestroyWindow("display");
	cvDestroyWindow("patch");
	cvDestroyWindow("prob_map");
}

int main(int argc, char **argv){
	string proto_path, weights_path, mean_path, video_path, init_bbox_path;
	proto_path = (argc>1)?argv[1]:"models/sodlt_deploy.prototxt";
	weights_path = (argc>2)?argv[2]:"example_sodlt_iter_10000.caffemodel";
	mean_path = (argc>3)?argv[3]:"data/mean.binaryproto";
	//image = (argc>4)?argv[4]:"image.jpg";
	video_path = (argc>4)?argv[4]:"0001R.MP4";
	init_bbox_path = (argc>5)?argv[5]:"init_bbox.xml";

	Classifier classifier(proto_path, weights_path, mean_path);

	CvCapture *video = cvCreateFileCapture(video_path.c_str());

	CvMat *init_bbox = (CvMat*)cvLoad(init_bbox_path.c_str());

	track(video, init_bbox, classifier);

	/*cv::Mat img = cv::imread(image, -1);
	CHECK(!img.empty()) << "Unable to decode image " << image;
	cv::Mat prediction = classifier.Predict(img);

	namedWindow("display", WINDOW_AUTOSIZE);
	IplImage prob_map = prediction;
	cvThreshold(&prob_map, &prob_map, 0.4, 1, CV_THRESH_BINARY);
	cvShowImage("display", &prob_map);
	cvWaitKey(0);
	cvDestroyWindow("display");*/
	return 0;
}