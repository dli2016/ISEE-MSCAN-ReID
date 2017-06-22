/**
 * \file  isee_mscan_reid.cpp
 * \brief implement the interfaces.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cassert>

#include <vector>
#include <string>

#include <caffe/caffe.hpp>

using namespace std;
using namespace caffe;

#include "isee_mscan_reid.hpp"

namespace cripac {

#define REID_SQR(x) ((x)*(x))

/*Initialization*/
int ISEEReID::initialize(const ReIDParams& params) {
  // Input checking.
  if (params.proto_filename == NULL) {
    fprintf(stderr, "Error: proto_filename is NULL\n");
    return ReID_ILLEGAL_ARG;
  }
  if (params.weights_filename == NULL) {
    fprintf(stderr, "Error: weights_filename is NULL\n");
    return ReID_ILLEGAL_ARG;
  }
  if (params.input_width < 0 || 
      params.input_height < 0 || 
      params.input_num_channels < 0) {
    fprintf(stderr, "Error: BAD image information!\n");
    return ReID_ILLEGAL_ARG;
  }

  input_width_ = params.input_width;
  input_height_= params.input_height;
  input_num_channels_ = params.input_num_channels;
  gpu_index_ = params.gpu_index;

  // Set gpu.
  setDevice(gpu_index_);

  // Load the network.
  fprintf(stdout, "Loading protocol from %s...\n", params.proto_filename);
  net_.reset(new Net<float>(params.proto_filename, TEST));
  fprintf(stdout, "Loading caffemodel from %s...\n", params.weights_filename);
  net_->CopyTrainedLayersFrom(params.weights_filename);
  fprintf(stdout, "Managing I/O blobs...\n");
  vector<Blob<float> *> input_blobs = net_->input_blobs();
  if (input_blobs.size() == 0) {
    return ReID_NO_INPUT_BLOB;
  }

  return ReID_OK; 
}

/* Implement comparison. */
float ISEEReID::compare(const PedestrianInfo& pedestrianA,
                        const PedestrianInfo& pedestrianB) {
  // Set device firstly.
  setDevice(gpu_index_);
  // Error.
  float score = -999999.9f;
  fprintf(stderr, "Error: no use now, maybe comming soon ...\n");
  return score;
}

/* Calculate similarity using extracted features. */
float ISEEReID::compare(const float* featureA, const float* featureB) {
  // Input checking.
  assert(featureA != NULL);
  assert(featureB != NULL);

  float accumulation = 0.0f;
  for (int i=0; i < kFeatureDims; ++i) {
    float sub_sqr = REID_SQR(featureB[i] - featureA[i]);
    accumulation += sub_sqr;
  }

  return sqrt(accumulation);
}

// Get features.
const float* ISEEReID::getFeature(const PedestrianInfo& pedestiran) {
  // Input checking.
  assert(pedestiran.tracklet_len > 0);
  assert(pedestiran.tracklet_data != NULL);

  // Set device.
  setDevice(gpu_index_);

  // Get features.
  const char* layer_name = "fc1_body";
  int tracklet_len = pedestiran.tracklet_len;
  float** tracklet_data = pedestiran.tracklet_data;

  Blob<float> *input_blob = net_->input_blobs()[0];
  input_blob->Reshape(tracklet_len, 
                      input_num_channels_, 
                      input_height_,
                      input_width_);
  boost::shared_ptr<caffe::Blob<float> > output_blob = 
    net_->blob_by_name(layer_name);

  float *dst = input_blob->mutable_cpu_data();
  int data_size = input_height_ * input_width_ * input_num_channels_;
  for (int i = 0; i < tracklet_len; ++i) {
    memcpy(dst, tracklet_data[i], sizeof(float) * data_size);
    dst += data_size;
  }
  net_->Forward();

  const float* feature_temp = output_blob->cpu_data();

  return calMeanFeature(tracklet_len, feature_temp);
}

/* Release. */
void ISEEReID::release(void) {
  net_.reset();
}


/* Private functions. */
/**
 * Set device id.
 * TODO: validate the gpu index.
 * int num_devices = 0;
 * CUDA_CHECK(cudaGetDeviceCount(&num_devices));
 */
int ISEEReID::setDevice(int gpu_index) {
  if (gpu_index < 0) {
    Caffe::set_mode(Caffe::CPU);
  } else {
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(gpu_index);
    //Caffe::DeviceQuery();
  }

  /*
  if (gpu_index >= num_devices) {
    return ReID_ILLEGAL_ARG;
  }
  */  

  return ReID_OK;
}

// Calculate the mean value of the feature of a sequence.
const float* ISEEReID::calMeanFeature(const int& num, const float* feature) {
  float* mean_feature = new float[kFeatureDims];
  memset(mean_feature, 0, kFeatureDims*sizeof(float));
  for (int ni = 0; ni < num; ++ni) {
    const float* fp = feature + ni*kFeatureDims;
    for (int fi = 0; fi < kFeatureDims; ++fi) {
      mean_feature[fi] += fp[fi]/(float)num;
    }
  }
  return mean_feature;
}

}
