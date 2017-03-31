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
	void PredictN(const cv::Mat *imgs, 
		int num_imgs, cv::Mat *output_probs, cv::Mat *output_bboxes);
private:
	void SetMean(const string &mean_file);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

	void PreprocessN(const cv::Mat *imgs, 
		int num_imgs, std::vector<cv::Mat>* input_channels);

	shared_ptr< Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	int num_batches_;
	cv::Mat mean_;
};

#endif