/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_cripac_isee_alg_pedestrian_reid_MSCANFeatureExtracter */

#ifndef _Included_org_cripac_isee_alg_pedestrian_reid_MSCANFeatureExtracter
#define _Included_org_cripac_isee_alg_pedestrian_reid_MSCANFeatureExtracter
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     org_cripac_isee_alg_pedestrian_reid_MSCANFeatureExtracter
 * Method:    initialize
 * Signature: (Lorg/cripac/isee/alg/pedestrian/reid/MSCANFeatureExtracter/MscanReIDParams;)J
 */
JNIEXPORT jlong JNICALL Java_org_cripac_isee_alg_pedestrian_reid_MSCANFeatureExtracter_initialize
  (JNIEnv *, jobject, jobject);

/*
 * Class:     org_cripac_isee_alg_pedestrian_reid_MSCANFeatureExtracter
 * Method:    getMeanVal
 * Signature: (JI)F
 */
JNIEXPORT jfloat JNICALL Java_org_cripac_isee_alg_pedestrian_reid_MSCANFeatureExtracter_getMeanVal
  (JNIEnv *, jobject, jlong, jint);

/*
 * Class:     org_cripac_isee_alg_pedestrian_reid_MSCANFeatureExtracter
 * Method:    calSimilarity
 * Signature: (J[F[F)F
 */
JNIEXPORT jfloat JNICALL Java_org_cripac_isee_alg_pedestrian_reid_MSCANFeatureExtracter_calSimilarity
  (JNIEnv *, jobject, jlong, jfloatArray, jfloatArray);

/*
 * Class:     org_cripac_isee_alg_pedestrian_reid_MSCANFeatureExtracter
 * Method:    extractFeature
 * Signature: (J[[F[F)V
 */
JNIEXPORT void JNICALL Java_org_cripac_isee_alg_pedestrian_reid_MSCANFeatureExtracter_extractFeature
  (JNIEnv *, jobject, jlong, jobjectArray, jfloatArray);

/*
 * Class:     org_cripac_isee_alg_pedestrian_reid_MSCANFeatureExtracter
 * Method:    free
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_cripac_isee_alg_pedestrian_reid_MSCANFeatureExtracter_free
  (JNIEnv *, jobject, jlong);

#ifdef __cplusplus
}
#endif
#endif
