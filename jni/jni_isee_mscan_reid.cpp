
/**
 * \file  jni_isee_mscan_reid.cpp
 * \brief Implement java native interface.
 * 
 * \version 0.1 2017/06/16;
 * \author  da.li
 */

#include <cstdlib>
#include <cstdio>
#include <string>

#include "isee_mscan_reid.hpp"
#include "jni_isee_mscan_reid.h"
using namespace cripac;

// Initialization.
JNIEXPORT jlong JNICALL Java_org_cripac_isee_alg_pedestrian_reid_MSCANFeatureExtracter_initialize
  (JNIEnv *env, jobject obj, jobject j_params) {

  // Find class.
  jclass params_class = 
    env->FindClass("org/cripac/isee/alg/pedestrian/reid/MSCANFeatureExtracter$MscanReIDParams");
  if (params_class == NULL) {
    fprintf(stderr,
      "Error: Cannot find Java class: MSCANFeatureExtracter$MscanReIDParams");
    fflush(stdout), fflush(stderr);
    return (jlong)NULL;
  }
  
  // Get field.
  // Width.
  jfieldID width_field  = env->GetFieldID(params_class, "inputWidth", "I");
  // Height.
  jfieldID height_field = env->GetFieldID(params_class, "inputHeight", "I");
  // Number of channels.
  jfieldID channel_field= env->GetFieldID(params_class, "inputChannels", "I");
  // Gpu Index.
  jfieldID gpu_field = env->GetFieldID(params_class, "gpuIndex", "I");
  // Path of protobuf file.
  jfieldID pbpath_field = env->GetFieldID(params_class, "protoPath", 
    "Ljava/lang/String;");
  // Path of caffe model.
  jfieldID modelpath_field = env->GetFieldID(params_class, "modelPath",
    "Ljava/lang/String;");
  
  // Get values.
  ReIDParams params;
  // Width.
  params.input_width = env->GetIntField(j_params, width_field);
  // Height.
  params.input_height= env->GetIntField(j_params, height_field);
  // Number of channels.
  params.input_num_channels = env->GetIntField(j_params, channel_field);
  // Gpu index.
  params.gpu_index = env->GetIntField(j_params, gpu_field);
  // Layer name.
  // Note: we have specified a layer name inside the library.
  // Path of protobuf file.
  jstring jstring_pb_path = (jstring)(env->GetObjectField(j_params, pbpath_field));
  const int pb_len = env->GetStringUTFLength(jstring_pb_path);
  char* c_pb_path = new char[pb_len + 1];
  env->GetStringUTFRegion(jstring_pb_path, 0, pb_len, c_pb_path);
  c_pb_path[pb_len] = '\0';
  params.proto_filename = c_pb_path;
  // Path of caffe model.
  jstring jstring_model_path = (jstring)(env->GetObjectField(j_params, modelpath_field));
  const int model_len = env->GetStringUTFLength(jstring_model_path);
  char* c_model_path = new char[model_len + 1];
  env->GetStringUTFRegion(jstring_model_path, 0, model_len, c_model_path);
  c_model_path[model_len] = '\0';
  params.weights_filename = c_model_path;

  // Initialize the comparer.
  ISEEReID* reider = new ISEEReID;
  int ret = reider->initialize(params);

  delete[] c_pb_path;
  c_pb_path = NULL;
  delete[] c_model_path;
  c_model_path = NULL;

  if (ret < 0) {
    fprintf(stderr, "Error: The tracker initialization FAILED!\n");
    fflush(stdout), fflush(stderr);
    return (jlong)NULL;
  } else {
    return (jlong)reider;
  }
}

// Get Mean Value.
JNIEXPORT jfloat JNICALL Java_org_cripac_isee_alg_pedestrian_reid_MSCANFeatureExtracter_getMeanVal
  (JNIEnv *env, jobject obj, jlong handle, jint color) {
  // Pointer to reid comparer.
  ISEEReID* reider = (ISEEReID*)handle;
  if (reider == NULL) {
    return -99999.9f;
  }
  jfloat mean_val = reider->getMeanVal(color);

  return mean_val;
}

// Calculate similarity.
// We now dont use the attributes information.
JNIEXPORT jfloat JNICALL Java_org_cripac_isee_alg_pedestrian_reid_MSCANFeatureExtracter_calSimilarity
  (JNIEnv *env, jobject obj, jlong handle, jfloatArray j_featureA, jfloatArray j_featureB) {
  // Pointer to reid comparer.
  ISEEReID* reider = (ISEEReID*)handle;
  if (reider == NULL) {
    return -99999.9f;
  }
  // Get input.
  float* c_featureA = env->GetFloatArrayElements(j_featureA, nullptr);
  float* c_featureB = env->GetFloatArrayElements(j_featureB, nullptr);
  // Calculate dissimilarity.
  float dissimilarity = reider->compare(c_featureA, c_featureB);
  // Release.
  env->ReleaseFloatArrayElements(j_featureA, c_featureA, 0);
  env->ReleaseFloatArrayElements(j_featureB, c_featureB, 0);
  // Return.
  return dissimilarity;
}

// Get reid feature.
JNIEXPORT void JNICALL Java_org_cripac_isee_alg_pedestrian_reid_MSCANFeatureExtracter_extractFeature
  (JNIEnv *env, jobject obj, jlong handle, jobjectArray j_tracklet_data, jfloatArray j_feature) {
  ISEEReID* reider = (ISEEReID*)handle;
  if (reider == NULL) {
    return;
  }
  // Set data used in c.
  int len_tracklet = env->GetArrayLength(j_tracklet_data);
  float** c_tracklet_data = new float*[len_tracklet];
  for (int ti = 0; ti < len_tracklet; ++ti) {
    c_tracklet_data[ti] =
      env->GetFloatArrayElements((jfloatArray) env->GetObjectArrayElement(j_tracklet_data, ti), nullptr);
  }
  // Get feature.
  PedestrianInfo pedestrian;
  pedestrian.tracklet_len = len_tracklet;
  pedestrian.tracklet_data= c_tracklet_data;
  pedestrian.attributes = NULL; 
  const float* c_feature = reider->getFeature(pedestrian);
  env->SetFloatArrayRegion(j_feature, 0, env->GetArrayLength(j_feature), c_feature);
  // release.
  for (int ti = 0; ti < len_tracklet; ++ti) {
    env->ReleaseFloatArrayElements((jfloatArray) env->GetObjectArrayElement(j_tracklet_data, ti), c_tracklet_data[ti], 0);
  }
  delete[] c_tracklet_data;
  c_tracklet_data = NULL;
  
  if (c_feature) {
    delete[] c_feature;
    c_feature = NULL;
  }
  return;
}

// Free resources.
JNIEXPORT void JNICALL Java_org_cripac_isee_alg_pedestrian_reid_MSCANFeatureExtracter_free
  (JNIEnv *env, jobject obj, jlong handle) {
  ISEEReID* reider = (ISEEReID*)handle;
  if (reider) {
    reider->release();
  }
  delete reider;
  reider = NULL;
}
