/*
input:
- deploy prototxt
- weights
- mean file
- video to track
- bounding box ground truth

output:
- bounding box prediction
*/

#include <cstdio>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

#include "main.h"
#include "Classifier.h"
#include "NNTracker.h"

using namespace std;

const int TRACK_MAX_PROPOSALS = 8;
const int PATCH_WIDTH = 100;
const int PATCH_HEIGHT = 100;

Mat stitchDisplay;

void stitchImages(Mat *crops, Mat &display, Mat *bboxes, Mat *probs, int numCrops){
	IplImage ipl = IplImage(display);
	cvZero(&ipl);
	int grid = sqrt(TRACK_MAX_PROPOSALS)+1;
	int k = 0;
	char scoreStr[10];
	for(int i=0;i<grid;++i){
		for(int j=0;j<grid;++j){
			if(k==numCrops)return;
			Mat roi = display(Rect(i*PATCH_WIDTH, j*PATCH_HEIGHT, PATCH_WIDTH, PATCH_HEIGHT));
			Mat crop;
			resize(crops[k], crop, cvSize(PATCH_WIDTH, PATCH_HEIGHT));
			crop.copyTo(roi);
			Rect bbox = Rect(bboxes[k].at<float>(0,0)*2,
				bboxes[k].at<float>(0,1)*2,
				bboxes[k].at<float>(0,2)*2 - bboxes[k].at<float>(0,0)*2,
				bboxes[k].at<float>(0,3)*2 - bboxes[k].at<float>(0,1)*2);
			rectangle(roi, cvPoint(bbox.x, bbox.y), cvPoint(bbox.x+bbox.width, bbox.y+bbox.height), CV_RGB(0x00, 0xff, 0x00), 1, CV_AA);
			
			float prob = probs[k].at<float>(0,0);
			sprintf(scoreStr, "%f", prob);
			putText(roi, scoreStr, cvPoint(0, roi.size().height), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0xff, 0xff, 0xff));
			++k;
		}
	}
}

int main(int argc, char **argv){
	string proto_path, weights_path, mean_path, video_path, ground_truth_path;
	proto_path = (argc>1)?argv[1]:"models/sodlt_deploy.prototxt";
	weights_path = (argc>2)?argv[2]:"example_sodlt_iter_10000.caffemodel";
	mean_path = (argc>3)?argv[3]:"data/mean.binaryproto";
	video_path = (argc>4)?argv[4]:"0001R.MP4";
	ground_truth_path = (argc>5)?argv[5]:"bboxseq.txt";

	Classifier classifier(proto_path, weights_path, mean_path);

	VideoCapture video(video_path);

	ifstream ground_truth;
	ground_truth.open(ground_truth_path.c_str(), ios::in);

	ofstream prediction;
	prediction.open("prediction.txt", ios::out | ios::trunc);

	//track(video, init_bbox, classifier);

	//use the NNTracker class
	NNTracker tracker(classifier);
	Rect proposals[1];
	int startFrame, currentFrame, x, y, w, h;
	ground_truth>>startFrame>>x>>y>>w>>h;
	proposals[0] = Rect(x, y, w, h);
	float prob;
	Rect bbox = Rect(0,0,0,0);
	
	Mat frame;
	video>>frame;
	if(frame.empty())return -1;
    stitchDisplay = Mat(cvSize((sqrt(TRACK_MAX_PROPOSALS)+1)*PATCH_WIDTH, (sqrt(TRACK_MAX_PROPOSALS)+1)*PATCH_HEIGHT), frame.type());

	struct timeval t1,t2;
	double timeuse;
	namedWindow("display", WINDOW_AUTOSIZE);
	int frameCount = 0;
	bool success;
	while(true){
		video>>frame;
		cvtColor(frame, frame, CV_BGR2RGB);
		if(frame.empty())break;
		frameCount++;
		if(frameCount<startFrame)continue;
#ifdef TIMING
		gettimeofday(&t1,NULL);
#endif
		if(frameCount==startFrame)
		 	success = tracker.track(frame, proposals, 1, &prob, &bbox);
		else
			success = tracker.track(frame, proposals, 0, &prob, &bbox);
		if(success){
		    stitchImages(tracker.patches, stitchDisplay, tracker.bboxes, tracker.probs, tracker.num_patches);
		    imshow("score", stitchDisplay);
		    for(int i=0;i<tracker.num_patches;++i){
		    	rectangle(frame, tracker.contexts[i], CV_RGB(0xff, 0x00, 0x00), 1);
		    }
	    }
#ifdef TIMING
		gettimeofday(&t2,NULL);
		timeuse = (t2.tv_sec - t1.tv_sec)*1000.0 + (t2.tv_usec - t1.tv_usec)/1000.0;
		printf("Use Time:%fms\n",timeuse);
#endif

#ifdef TRACK_DEBUG_MODE
		rectangle(frame, Point(bbox.x, bbox.y),
			Point(bbox.x+bbox.width, bbox.y+bbox.height), CV_RGB(0x00, 0xff, 0x00), 1, CV_AA);
		rectangle(frame, Point(x, y),
			Point(x+w, y+h), CV_RGB(0x00, 0x00, 0xff), 1, CV_AA);
		imshow("display", frame);
		char c = waitKey(0);
		if(c==27)break;
#endif
		if(ground_truth.eof())break;
		ground_truth>>currentFrame>>x>>y>>w>>h;
		prediction<<frameCount<<' '<<bbox.x<<' '<<bbox.y<<' '<<bbox.width<<' '<<bbox.height<<endl;
	}
	destroyWindow("display");
	ground_truth.close();
	prediction.close();

	return 0;
}