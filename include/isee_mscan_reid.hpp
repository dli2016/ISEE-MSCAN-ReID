/**
 * @file    isee_mscan_reid.h
 * @brief   The interface to do reid by calling the trained caffe model.
 *          (the model is provided by Dangwei Li)
 * @version 0.1 on 2017/06/13  0.2 on 2017/06/21
 * @author  by da.li
 */

#ifndef _ISEE_MSCAN_REID_HPP_
#define _ISEE_MSCAN_REID_HPP_

#include <memory>
#include <boost/shared_ptr.hpp>

#include "attributes.h"

namespace caffe {
template<typename Dtype>
class Net;
template<typename Dtype>
class Blob;
}

namespace cripac{
/**
 * @struct ReIDParams
 */
typedef struct reid_params_t_ {

  char* proto_filename;
  char* weights_filename;
  int input_width;
  int input_height;
  int input_num_channels;
  int gpu_index;

} ReIDParams;

/**
 * @struct PedestrianInfo
 * @brief  the color data should be stored with the format B...B...B,
 *         G...G...G, R...R...R and row domain; the size of tracklet
 *         data is tracklet_len x num_channels x width x height.
 */
typedef struct pedestrian_info_t_ {

  int tracklet_len;
  float** tracklet_data;
  Attributes* attributes;

} PedestrianInfo;

/**
 * @class ISEEReID
 * @brief For this version, we only use the appearance features to
 *        achieve our purpose; but in the interface we also set the
 *        variable to input the attributes for every frames of the
 *        tracklet (now we leave it NULL).
 */
class ISEEReID {

 public:
  /*Constructor & Destructor*/
  ISEEReID(void) {
    input_width_ = 0;
    input_height_= 0;
    input_num_channels_ = 0;
    gpu_index_ = -1;
    _setInnerVariables();
  }
  ~ISEEReID(void) {
    release();
  }

  /* Dimensions of the feature. */
  const static int kFeatureDims = 128;

  /**
   * @enum  ISEEReIDStatus
   * @brief Error types.
   */
  enum ISEEReIDStatus {
    ReID_OK = 0,
    ReID_ILLEGAL_ARG = -1,
    ReID_NO_INPUT_BLOB = -2,
  };

  /**
   * @enum Color.
   */
  enum Color {
    B = 0,
    G = 1,
    R = 2,
  };

  /**
   * Initialization
   * \param[IN] params - necessary parameters.
   *            (more details see the struct ReIDParams).
   * \return    Error types.
   */
  int initialize(const ReIDParams& params);
  
  /**
   * Call the trained model to generate the features.
   * Note: you must release the feature memory after using it.
   * \param[IN] pedestrian (gbr data and length of its tracklet).
   * \return get the feature from trained caffe model.
   */
  const float* getFeature(const PedestrianInfo& pedestrian);

  /**
   * Calculate the similariry of the two input tracklets.
   * This interface to get the similarity by the trained caffe model directly.
   * \param[IN] PedestrianA
   * \param[IN] PedestrianB
   * \return the similarity.
   */
  float compare(const PedestrianInfo& pedestrianA, 
                const PedestrianInfo& pedestrianB);

  /**
   * Calculate the similariry using extracted features.
   * \param[IN] feature of pedestrian A.
   * \param[IN] feature of pedestrian B.
   * \return L2 distance of the two input features.
   */
  float compare(const float* featureA, const float* featureB);

  /**
   * Release the resources.
   */
  void release(void);

  /**
   * Get mean value..
   */
  float getMeanVal(int color) {
    if (B == color) {
      return _mean_value_color_B;
    } else if (G == color) {
      return _mean_value_color_G;
    } else if (R == color) {
      return _mean_value_calor_R;
    } else {
      return (float)ReID_ILLEGAL_ARG;
    }
  }
  
  /**
   * Get scale.
   */
  float getScale(void) {
    return _scale;
  }

 private:
  /**
   * Set device.
   */
  int setDevice(int gpu_index);

  /**
   * Calculate the mean feature.
   */ 
  const float* calMeanFeature(const int& num, const float* feature);

  /**
   * Set the values of inner variables.
   */ 
  void _setInnerVariables(void) {
    _mean_value_color_B = 97.5742f;
    _mean_value_color_G = 98.7142f;
    _mean_value_calor_R = 105.2910f;
    _scale = 0.00390625f;
  }

  /**
   * Memember Variables.
   */
  int input_width_;
  int input_height_;
  int input_num_channels_;
  int gpu_index_;
  std::shared_ptr<caffe::Net<float> > net_;

  /**
   * Inner Variables.
   */
  float _mean_value_color_B;
  float _mean_value_color_G;
  float _mean_value_calor_R;
  float _scale;
  
};

}

#endif  // _ISEE_MSCAN_REID_HPP_
