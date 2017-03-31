#include <caffe/caffe.hpp>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>
#include "Classifier.h"

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file) {
	Caffe::set_mode(Caffe::GPU);
	Caffe::SetDevice(0);
	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	//CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
  /*batch size*/
  num_batches_ = input_layer->shape(0);
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
	<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
	/* Load the binaryproto mean file. */
	SetMean(mean_file);

	Blob<float>* output_layer = net_->output_blobs()[0];
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->shape(0) * input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::PreprocessN(const cv::Mat *imgs, int num_imgs, std::vector<cv::Mat>* input_channels) {
  int num_inputs = (num_imgs > num_batches_)?num_batches_:num_imgs;
  for(int i=0;i<num_inputs;++i){
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (imgs[i].channels() == 3 && num_channels_ == 1)
      cv::cvtColor(imgs[i], sample, cv::COLOR_BGR2GRAY);
    else if (imgs[i].channels() == 4 && num_channels_ == 1)
      cv::cvtColor(imgs[i], sample, cv::COLOR_BGRA2GRAY);
    else if (imgs[i].channels() == 4 && num_channels_ == 3)
      cv::cvtColor(imgs[i], sample, cv::COLOR_BGRA2BGR);
    else if (imgs[i].channels() == 1 && num_channels_ == 3)
      cv::cvtColor(imgs[i], sample, cv::COLOR_GRAY2BGR);
    else
      sample = imgs[i];

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
      cv::resize(sample, sample_resized, input_geometry_);
    else
      sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
      sample_resized.convertTo(sample_float, CV_32FC3);
    else
      sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, *(input_channels+i*num_channels_));

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
      << "Input channels are not wrapping the input layer of the network.";
  }
}

void Classifier::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

void Classifier::PredictN(const cv::Mat *imgs, int num_imgs, cv::Mat *output_probs, cv::Mat *output_bboxes) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(num_batches_, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  PreprocessN(imgs, num_imgs, &input_channels);

  net_->Forward();

  int num_outputs = (num_imgs > num_batches_)?num_batches_:num_imgs;

  /* Copy the output layer to a std::vector */
  CHECK(net_->output_blobs().size() == 2)
    << "Output should contains probability and bounding box";
  Blob<float>* output_layer_prob = net_->output_blobs()[1];
  Blob<float>* output_layer_bbox = net_->output_blobs()[0];
  /*probability output should be a single value*/
  if(output_layer_prob->count(1)!=1){
    output_layer_prob = net_->output_blobs()[0];
    output_layer_bbox = net_->output_blobs()[1];
  }
  int out_width_prob = output_layer_prob->channels();
  int out_width_bbox = output_layer_bbox->channels();
  float* output_data_bbox = output_layer_bbox->mutable_cpu_data();
  float* output_data_prob = output_layer_prob->mutable_cpu_data();
  for(int i=0;i<num_outputs;++i){
    cv::Mat output_prob(1, out_width_prob, CV_32FC1, output_data_prob);
    output_probs[i] = output_prob;
    output_data_prob += out_width_prob;
    cv::Mat output_bbox(1, out_width_bbox, CV_32FC1, output_data_bbox);
    output_bboxes[i] = output_bbox;
    output_data_bbox += out_width_bbox;
  }
}

cv::Mat Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  if(net_->output_blobs().size()>1){
    /*bbox output should be a matrix*/
    if(output_layer->count(1)==1)
      output_layer = net_->output_blobs()[1];
  }
  int out_channels = output_layer->channels();
  float* output_data = output_layer->mutable_cpu_data();
  cv::Mat output(1, out_channels, CV_32FC1, output_data);
  return output;
}
