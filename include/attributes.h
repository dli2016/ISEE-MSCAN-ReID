/**
 * \file  attributes.h
 * \brief The score of each attribute. The attributes in the class are
 *        labeled in RAP dataset.
 */

#ifndef _ATTRIBUTES_H_
#define _ATTRIBUTES_H_

typedef struct attributes_t_ {
  // Gender.
  float gender_male;
  float gender_female;
  float gender_other;

  // Age.
  float age_16;
  float age_30;
  float age_45;
  float age_60;
  float age_older_60;

  // Weight.
  float weight_very_fat;
  float weight_little_fat;
  float weight_normal;
  float weight_little_thin;
  float weight_very_thin;

  // Role.
  float role_client;
  float role_uniform;

  // Hair.
  float hair_style_null;
  float hair_style_long;

  // head shoulder.
  float head_shoulder_black_hair;
  float head_shoulder_with_hat;
  float head_shoulder_glasses;
  float head_shoulder_sunglasses;
  float head_shoulder_scarf;
  float head_shoulder_mask;

  // Upper.
  float upper_shirt;
  float upper_sweater;
  float upper_vest;
  float upper_tshirt;
  float upper_cotton;
  float upper_jacket;
  float upper_suit;
  float upper_hoodie;
  float upper_cotta;
  float upper_other;
  float upper_black;
  float upper_white;
  float upper_gray;
  float upper_red;
  float upper_green;
  float upper_blue;
  float upper_silvery;
  float upper_yellow;
  float upper_brown;
  float upper_purple;
  float upper_pink;
  float upper_orange;
  float upper_mix_color;
  float upper_other_color;
  
  // Lower.
  float lower_pants;
  float lower_short_pants;
  float lower_skirt;
  float lower_short_skirt;
  float lower_long_skirt;
  float lower_one_piece;
  float lower_jean;
  float lower_tight_pants;
  float lower_black;
  float lower_white;
  float lower_gray;
  float lower_red;
  float lower_green;
  float lower_blue;
  float lower_silver;
  float lower_yellow;
  float lower_brown;
  float lower_purple;
  float lower_pink;
  float lower_orange;
  float lower_mix_color;
  float lower_other_color;

  // Shoes.
  float shoes_leather;
  float shoes_sport;
  float shoes_boot;
  float shoes_cloth;
  float shoes_shandle;
  float shoes_casual;
  float shoes_other;
  float shoes_black;
  float shoes_white;
  float shoes_gray;
  float shoes_red;
  float shoes_green;
  float shoes_blue;
  float shoes_silver;
  float shoes_yellow;
  float shoes_brown;
  float shoes_purple;
  float shoes_pink;
  float shoes_orange;
  float shoes_mix_color;
  float shoes_other_color;
  
  // Accessory.
  float accessory_backpack;
  float accessory_shoulderbag;
  float accessory_handbag;
  float accessory_waistbag;
  float accessory_box;
  float accessory_plasticbag;
  float accessory_paperbag;
  float accessory_cart;
  float accessory_kid;
  float accessory_other;

  // Action.
  float action_calling;
  float action_armstretching;
  float action_chatting;
  float action_gathering;
  float action_lying;
  float action_crouching;
  float action_running;
  float action_holdthing;
  float action_pushing;
  float action_pulling;
  float action_nipthing;
  float action_picking;
  float action_other;

  // View Angle.
  float view_angle_left;
  float view_angle_right;
  float view_angle_front;
  float view_angle_back;

  // Occlusion.
  float occlusion_left;
  float occlusion_right;
  float occlusion_up;
  float occlusion_down;
  float occlusion_environment;
  float occlusion_accessory;
  float occlusion_object;
  float occlusion_other;

} Attributes; // _ATTRIBUTES_H_

#endif
