
/**
 * \file main.cpp
 */

#include <stdlib.h>
#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>

#include <opencv2/opencv.hpp>
using namespace cv;

#include <string>
#include <vector>
using namespace std;

#include "isee_mscan_reid.hpp"
using namespace cripac;

typedef vector<Mat> ImageSet;
typedef vector<String> FilenameSet;
typedef unsigned long DWORD;
/**
 * test stub.
 */
// Data convert.
int dataConvert(const ImageSet& image, float** data);
// Normalization.
int dataNormalize(float* data);
// Get file paths.
int getFilenames(const string& dir, FilenameSet& filenames);
// Load images.
int loadImages(const FilenameSet& image_files, float** data);
// Release 2dims array.
void release2DArray(float** data, int outside);
// Record time.
DWORD GetTickCount() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

enum TestStubErrType {
  TEST_STUB_OK = 0,
  BAD_NUM_INPUTS = -10,
  LOAD_FILENAME_ERROR = -11,
  LOAD_IMAGE_ERROR = -12,
};

/* Mean values. */
const static float kMeanColorB = 97.5742f;
const static float kMeanColorG = 98.7142f;
const static float kMeanColorR = 105.2910f;

const static float kScale = 1.0f / 256.0f;

const static int kNumChannels = 3;
const static int kInputHeight = 160;
const static int kInputWidth = 64;
/**
 * Main function.
 */
int main(int argc, char** argv) {
  // Check...
  if (argc != 3) {
    fprintf(stderr, "Error: BAD number of input arguments\n");
    return BAD_NUM_INPUTS;
  }
  string proto_filename = "models/deploy.prototxt";
  string caffe_model = "models/mscan.caffemodel";
  
  string pedestrian_dir_A = argv[1];
  string pedestrian_dir_B = argv[2];

  // Load filename.
  printf("==== Get the filenames of test samples.\n");
  FilenameSet filesetA;
  FilenameSet filesetB;
  int ret = getFilenames(pedestrian_dir_A, filesetA);
  if (ret < 0) {
    filesetA.clear();
    return ret;
  }
  ret = getFilenames(pedestrian_dir_B, filesetB);
  if (ret < 0) {
    filesetA.clear();
    filesetB.clear();
    return ret;
  }
  printf("==== Get the filenames of  test samples DONE!\n");

  // Load images (convert them).
  printf("==== Load test images ... And data format convert.\n");
  int input_image_size = kInputWidth*kInputHeight*kNumChannels;
  // Set A.
  int num_filesetA = (int)filesetA.size();
  float** dataA = new float*[num_filesetA];
  for (int i = 0; i < num_filesetA; ++i) {
    dataA[i] = new float[input_image_size];
  }
  ret = loadImages(filesetA, dataA);
  if (ret < 0) {
    release2DArray(dataA, num_filesetA);
    filesetA.clear();
    filesetB.clear();
    return ret;
  }
  //for (int i = 0; i < num_filesetA; ++i) {
  //  printf("00 %f\n", dataA[i][2*kInputWidth*kInputHeight + 1000]/kScale + kMeanColorR);
  //}
  int num_filesetB = (int)filesetB.size();
  float** dataB = new float*[num_filesetB];
  for (int i = 0; i < num_filesetB; ++i) {
    dataB[i] = new float[input_image_size];
  }
  ret = loadImages(filesetB, dataB);
  if (ret < 0) {
    release2DArray(dataA, num_filesetA);
    release2DArray(dataB, num_filesetB);
    filesetA.clear();
    filesetB.clear();
    return ret;
  }
  printf("==== Load test images and data format converting DONE!\n");

  // Compare.
  printf("==== REID NOW ====\n");
  printf("  -- Create ISEEReID object\n");
  ISEEReID *reid = new ISEEReID;

  ReIDParams params;
  string layer_name = "fc1_body";
  params.proto_filename = (char*)proto_filename.c_str();
  params.weights_filename = (char*)caffe_model.c_str();
  params.input_width = kInputWidth;
  params.input_height = kInputHeight;
  params.input_num_channels = kNumChannels;
  params.gpu_index = 0;
  printf("  -- ISEEReID initialization ...\n");
  ret = reid->initialize(params);
  if (ret < 0) {
    fprintf(stderr, "Error: ISEEReID initialization FAILED, error code %d\n",
      ret);
    release2DArray(dataA, num_filesetA);
    release2DArray(dataB, num_filesetB);
    filesetA.clear();
    filesetB.clear();
    if (reid) {
      delete reid;
      reid = NULL;
    }
    return ret;
  }
  printf("  -- ISEEReID initialize successfully!\n");

  // start compare.
  printf("  -- Start compare...\n");
  PedestrianInfo person_A;
  PedestrianInfo person_B;
  person_A.tracklet_len = num_filesetA;
  person_A.tracklet_data = dataA;
  person_A.attributes = NULL;
  person_B.tracklet_len = num_filesetB;
  person_B.tracklet_data = dataB;
  person_B.attributes = NULL;
  DWORD start_time, end_time;

  int cnt = 0;
  int total = 10000;
  DWORD start = GetTickCount();
  while (cnt < total) {
    cnt++;
    printf(" == Round %d == \n", cnt);
    start_time = GetTickCount();
    const float* featureA = reid->getFeature(person_A);
    const float* featureB = reid->getFeature(person_B);
    float score = reid->compare(featureA, featureB);
    if (featureA) {
      delete[] featureA;
      featureA = NULL;
    }
    if (featureB) {
      delete[] featureB;
      featureB = NULL;
    }
    end_time = GetTickCount();
    printf("  -- Tracklet length of person A is %d\n", num_filesetA);
    printf("  -- Tracklet lentth of person B is %d\n", num_filesetB);
    printf("  -- dissimilarity of the two person is %f\n", score);
    printf("  -- The elapsed time is %d ms\n", end_time - start_time);
  }
  DWORD end = GetTickCount();
  printf("Total elapsed time is %4f s\n", (end - start) / 1000.0f);

  // release.
  filesetA.clear();
  filesetB.clear();

  release2DArray(dataA, num_filesetA);
  release2DArray(dataB, num_filesetB);

  if (reid) {
    delete reid;
    reid = NULL;
  }

  return TEST_STUB_OK;
}

//====================== Test Stub Implementation =======================
void release2DArray(float** data, int outside) {
  for (int i = 0; i < outside; ++i) {
    if (data[i]) {
      delete[] data[i];
      data[i] = NULL;
    }
  }
  delete[] data;
  data = NULL;
}

int dataConvert(const Mat& image, float* data) {
  Mat channels[kNumChannels];
  split(image, channels);

  float mean_val[kNumChannels] = {kMeanColorB, kMeanColorG, kMeanColorR};

  //Mat B = channels[0];
  //Mat G = channels[1];
  //Mat R = channels[2];

  //printf("(%f, %f)\n", (float)G.data[64*160-1], (float)image.data[3*64*160-2]);
  int image_size = kInputHeight * kInputWidth;
  for (int ci=0; ci < kNumChannels; ++ci) {
    for (int pi=0; pi < kInputHeight*kInputWidth; ++pi) {
      data[ci*image_size + pi] = (channels[ci].data[pi] - mean_val[ci]) * kScale;
    }
  }
  //int check_index = 2;
  //printf("(%f, %f)\n", (float)R.data[1000], 
  //  data[check_index*image_size + 1000]/kScale + mean_val[check_index]);
 
  return TEST_STUB_OK;
}

int dataNormalize(float* data) {
  return TEST_STUB_OK;
}

int loadImages(const FilenameSet& image_files, float** data) {
  int num_files = (int)image_files.size();
  //namedWindow( "Display window", CV_WINDOW_NORMAL );
  //ImageSet images;
  Size size;
  size.height = kInputHeight;
  size.width = kInputWidth;
  for (int i = 0; i < num_files; ++i) {
    Mat image;
    image = imread(image_files[i].c_str(), CV_LOAD_IMAGE_COLOR);
    if (!image.data) {
      fprintf(stderr, "Error: Load the %d image FAILED\n", i);
      //images.clear();
      return LOAD_IMAGE_ERROR;
    } else {
      //image.convertTo(image, CV_32FC3);
      resize(image, image, size);
      //image.convertTo(image, CV_32FC3);
      //images.push_back(image_resize);
      //imshow( "Display window", image );
      //waitKey(0);
      dataConvert(image, data[i]);
    }
  }

  return TEST_STUB_OK;
}

int getFilenames(const string& input_dir, FilenameSet& filenames) {
  filenames.clear();
  const int kTypeFile = 8;
  const int kBaseLen = 1000;

  DIR * dir;
  struct dirent * filename;

  // Check input dir is validate.
  struct stat s;
  lstat(input_dir.c_str(), &s);
  if ( !S_ISDIR( s.st_mode )) {
    fprintf(stderr, "Error: input dir name is NOT validate!\n");
    return LOAD_FILENAME_ERROR;
  }

  dir = opendir( input_dir.c_str() );
  if (NULL == dir) {
    fprintf(stderr,"Error: Open dir FAILED!\n");
    return LOAD_FILENAME_ERROR;
  }

  while ((filename = readdir(dir)) != NULL) {
    if(strcmp( filename->d_name , "." ) == 0 ||   
       strcmp( filename->d_name , "..") == 0) {
      continue;
    } else if (filename->d_type == kTypeFile) {
      string filename_tmp = filename->d_name;
      filename_tmp = input_dir + "/" + filename_tmp;
      filenames.push_back(filename_tmp);
    }
  }
  closedir(dir);
  dir = NULL;

  return TEST_STUB_OK;
}
