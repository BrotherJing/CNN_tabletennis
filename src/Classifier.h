#ifndef HEADER_CLASSIFIER
#define HEADER_CLASSIFIER

using namespace caffe;
using namespace cv;
using namespace std;

class Classifier{
public:
	Classifier(const string &model_file,
             const string& trained_file,
             const string& mean_file);
	cv::Mat Predict(const cv::Mat &img);
private:
	void SetMean(const string &mean_file);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);
	shared_ptr< Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
};

#endif