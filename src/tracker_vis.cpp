/*
input:
- video to track
- bounding box ground truth
- other predicted bboxes

output:
- bounding box prediction
*/

#include <cstdio>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

#include "main.h"

double SCALE = 0.5;

using namespace std;
using namespace cv;

CvScalar colors[3] = {CV_RGB(0xff, 0x00, 0x00),CV_RGB(0xff, 0x00, 0xff),CV_RGB(0xff, 0xff, 0x00)};
// 0:red 1:pink 2:yellow

int main(int argc, char **argv){
	string video_path, ground_truth_path;
	vector<string> predictions;
	video_path = (argc>1)?argv[1]:"0001R.MP4";
	ground_truth_path = (argc>2)?argv[2]:"bboxseq.txt";
	for(int i=3;i<argc;++i){
		predictions.push_back(string(argv[i]));
	}
	if(predictions.size()==0){
		cout<<"no prediction file."<<endl;
		return 0;
	}

	VideoCapture video(video_path);

	ifstream ground_truth;
	ground_truth.open(ground_truth_path.c_str(), ios::in);

	ifstream preds[predictions.size()];
	for(int i=0;i<predictions.size();++i){
		preds[i].open(predictions[i].c_str(), ios::in);
	}

	int startFrame, currentFrame, x, y, w, h;
	ground_truth>>startFrame>>x>>y>>w>>h;
	
	Mat frame, frameDisplay;
	video>>frame;
	if(frame.empty())return -1;
	Size small = Size(frame.cols/SCALE, frame.rows/SCALE);
    
    namedWindow("display", WINDOW_AUTOSIZE);
	int frameCount = 0;
	bool success;
	while(true){
		video>>frame;
		if(frame.empty())break;
		cvtColor(frame, frame, CV_BGR2RGB);
		resize(frame, frameDisplay, small);
		frameCount++;
		if(frameCount<startFrame)continue;

		for(int j=0;j<predictions.size();++j){
			int currentFrame1,x1,y1,w1,h1;
			preds[j]>>currentFrame1>>x1>>y1>>w1>>h1;
			x1/=SCALE;
			y1/=SCALE;
			w1/=SCALE;
			h1/=SCALE;
			rectangle(frameDisplay, Point(x1,y1), Point(x1+w1,y1+h1), colors[j], 1);
		}

		rectangle(frameDisplay, Point(x/SCALE, y/SCALE), Point((x+w)/SCALE, (y+h)/SCALE), CV_RGB(0x00, 0xff, 0x00), 1, CV_AA);
		imshow("display", frameDisplay);
		char c = waitKey(0);
		if(c==27)break;

		if(ground_truth.eof())break;
		ground_truth>>currentFrame>>x>>y>>w>>h;
	}
	destroyWindow("display");
	ground_truth.close();
	for(int i=0;i<predictions.size();++i)preds[i].close();

	return 0;
}