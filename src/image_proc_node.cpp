#include "camera_calibration_virtana/image_proc.h"

int main(int argc, char** argv) {

  ros::init(argc, argv, "image_proc_node");

  camera_calibration_virtana::ImageProcessor img_proc;

  ROS_INFO("Spinning node");
  ros::spin();

  return 0;
}
